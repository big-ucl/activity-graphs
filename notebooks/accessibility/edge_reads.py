"""Detection reads of message passing over the rail edges (accessibility plan, step 6), read from a recorded run.

Run with ``uv run python notebooks/accessibility/edge_reads.py <data> <run>`` to print every table of one run of the
``edge_ablation`` experiment. Each read compares a model with the arms of its family that pass messages over fewer
edges: ``-norail`` (the contiguity edges alone) and ``-nocontig-norail`` (no edge).

- The zone split: the paired per-user ``avg_recall`` split by the zone of each positive: hub zones, other station
  zones and zones with no station. A station zone has a rail edge. A hub zone is one of the top 5% of the station zones
  by rail degree, the number of zones a rail edge joins it to. When more zones share the degree of the last hub than
  the top 5% holds, none of the zones at that degree is a hub.
- The rank tail: the paired mean log2 rank of the test positives, split by the rank of each positive under
  the ``MLP`` variant of the run, outside the comparison, and, beyond rank ``TAIL_RANK`` under it, by hop band and by zone. A
  negative difference ranks the positives higher than the reference. The test pair accuracy of every arm is printed
  beside it.
- The embedding similarity: the mean cosine similarity of the node embeddings after the message-passing layers,
  between station zones joined by a rail edge and between station zones not joined by one. Two station zones joined by
  a rail edge are served by one rail trip.
- The attention: for each GATv2 layer and each zone class, the mean entropy of the attention of a zone over its
  neighbours, divided by its largest value, and the shares of the attention on the zone itself, on its contiguity
  neighbours and on its other rail neighbours.

The GAT reads embed ``N_EMBEDDING_USERS`` test users, one graph copy each. The two-tower node tower embeds the zones
once, with no user. The model reads run on the CPU, so they can run beside a training queue.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from activitygraphs.analysis import aggregate_metrics, paired_comparison, read_report_run
from activitygraphs.base import PER_USER_STAGE
from activitygraphs.config import Config
from activitygraphs.datasets import HouseholdSplit, Locations, SurveyDataModule, UserBatch, UserRows
from activitygraphs.experiments.variants import build_variants
from activitygraphs.models.architectures import build_scorer
from activitygraphs.models.networks import GAT
from activitygraphs.models.scorers import GraphScorer, Scorer
from activitygraphs.models.two_tower import TwoTowerScorer
from activitygraphs.runs import CHECKPOINTS_DIR, CONFIG_FILE, run_dir
from activitygraphs.schema import DatasetSchema, EdgeKind
from activitygraphs.training.metrics import DEFAULT_HOP_BANDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARM_SUFFIXES = ("", "-norail", "-nocontig-norail")
REFERENCE_SUFFIXES = ("-norail", "-nocontig-norail")
SEEDS = (42, 43, 44)

HUB_SHARE = 0.05
TAIL_RANK = 50
GROUPING_RANK_BANDS = (("1-50", 1.0, 50.0), ("51-200", 50.0, 200.0), ("201+", 200.0, float("inf")))
ZONE_CLASSES = ("hub", "station", "no station")
N_EMBEDDING_USERS = 32
N_BOOTSTRAP = 2000


# =========================================
# Zones
# =========================================


def rail_degree(rail_edge_index: Tensor, num_nodes: int) -> np.ndarray:
    """Return the rail degree of every node: the number of nodes a rail edge joins it to.

    Args:
        rail_edge_index: Rail edges in both directions, each ordered pair of nodes once, ``[2, num_rail_edges]``.
        num_nodes: Number of nodes of the graph.
    """
    return np.bincount(rail_edge_index[0].cpu().numpy(), minlength=num_nodes)


def zone_classes(degree: np.ndarray, hub_share: float) -> np.ndarray:
    """Return the class of every node: ``"hub"``, ``"station"`` or ``"no station"``.

    A station node has a rail degree above 0. The hubs are the top ``hub_share`` of the station nodes by rail degree,
    rounded, and at least one. When more nodes share the degree of the last hub than that count holds, none of the
    nodes at that degree is a hub.

    Args:
        degree: Rail degree of every node, from ``rail_degree``.
        hub_share: Share of the station nodes that are hubs.
    """
    is_station = degree > 0
    num_hubs = max(1, round(hub_share * int(is_station.sum())))
    cutoff = np.sort(degree[is_station])[::-1][num_hubs - 1]
    is_hub = degree > cutoff if (degree >= cutoff).sum() > num_hubs else degree >= cutoff

    classes = np.full(len(degree), ZONE_CLASSES[2], dtype=object)
    classes[is_station] = ZONE_CLASSES[1]
    classes[is_hub] = ZONE_CLASSES[0]

    return classes


# =========================================
# Reads from the saved ranks
# =========================================


def arm_families(names: Sequence[str]) -> list[str]:
    """Return the variants of ``names`` that have a ``-norail`` arm and a ``-nocontig-norail`` arm in ``names``, sorted."""
    name_set = set(names)
    return sorted(name for name in name_set if all(name + suffix in name_set for suffix in REFERENCE_SUFFIXES))


def grouping_model(names: Sequence[str]) -> str:
    """Return the ``MLP`` variant of ``names``, to group the positives of the rank tail by.

    Raises:
        ValueError: ``names`` holds no ``MLP`` variant, or more than one.
    """
    mlp_names = [name for name in names if name.startswith("MLP")]
    if len(mlp_names) != 1:
        raise ValueError(f"The run needs exactly one `MLP` variant to group the positives by, found {mlp_names}.")

    return mlp_names[0]


def positive_frame(per_user: pl.DataFrame, classes: np.ndarray) -> pl.DataFrame:
    """Return one row per test positive of every model and seed, with its rank, its hop band and its zone class.

    Args:
        per_user: Per-user results with the ranking metrics, from ``read_report_run``.
        classes: Class of every node, from ``zone_classes``.

    Returns:
        Frame of ``name, seed, user_idx, n_pos, pos_node, hop_band, zone_class, pos_avg_recall, rank, log2_rank``.
        ``rank`` is the expected rank of the positive among the candidates of its user, ties broken at random.
    """
    rank_columns = ["pos_node", "pos_hops", "pos_n_scored_higher", "pos_n_tied", "pos_avg_recall"]
    positives = (
        per_user
        .filter(pl.col("stage") == PER_USER_STAGE)
        .with_columns(n_pos=pl.col("pos_node").list.len())
        .select("name", "seed", "user_idx", "n_pos", *rank_columns)
        .explode(rank_columns)
    )
    node_classes = pl.DataFrame({"pos_node": np.arange(len(classes)), "zone_class": classes.astype(str)})
    rank = pl.col("pos_n_scored_higher") + pl.col("pos_n_tied") / 2 + 1

    return (
        positives
        .join(node_classes, on="pos_node")
        .with_columns(hop_band=_hop_band(pl.col("pos_hops")), rank=rank, log2_rank=rank.log(2))
        .drop("pos_hops", "pos_n_scored_higher", "pos_n_tied")
    )


def paired_zone_split(positives: pl.DataFrame, reference: str) -> pl.DataFrame:
    """Return the paired per-user ``avg_recall`` of every other model against ``reference``, split by zone class.

    The value of a user in a class is the contribution of the user's positives in that class to the user's
    ``avg_recall``, so the classes of a user sum to the user's ``avg_recall``.

    Args:
        positives: Positives of the models to compare, from ``positive_frame``.
        reference: Name of the reference model.

    Returns:
        Frame of ``zone_class, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``.
    """
    contributions = positives.group_by("name", "seed", "user_idx", "zone_class").agg(
        contribution=pl.col("pos_avg_recall").sum() / pl.col("n_pos").first()
    )

    return _paired_by_group(contributions, reference, "contribution", "zone_class", ZONE_CLASSES)


def grouping_ranks(positives: pl.DataFrame, model: str) -> pl.DataFrame:
    """Return the rank of every test positive under ``model``, averaged over its seeds, to group the positives by.

    Returns:
        Frame of ``user_idx, pos_node, grouping_rank``, one row per positive.
    """
    return (
        positives
        .filter(pl.col("name") == model)
        .group_by("user_idx", "pos_node")
        .agg(grouping_rank=pl.col("rank").mean())
    )


def paired_rank_tail(positives: pl.DataFrame, reference: str, ranks: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the paired mean log2 rank of the positives of every other model against ``reference``.

    The positives are grouped by ``ranks``, their rank under a model outside the comparison, so that the grouping does
    not favour either side. A grouping by the rank under ``reference`` would: a positive ranked far down by the reference
    is partly there by chance, so any other model ranks it higher on average. The value of a user in a group is the
    mean ``log2_rank`` of its positives in the group.

    Args:
        positives: Positives of the models to compare, from ``positive_frame``.
        reference: Name of the reference model.
        ranks: Rank of every positive to group by, from ``grouping_ranks``.

    Returns:
        The paired comparisons, each a frame of ``group, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p,
        per_seed_sd``, keyed by ``"grouping rank"`` (every positive), ``"hop band"`` and ``"zone class"`` (the
        positives beyond ``TAIL_RANK`` in ``ranks``).
    """
    ranked = positives.join(ranks, on=["user_idx", "pos_node"]).with_columns(
        grouping_band=_rank_band(pl.col("grouping_rank"))
    )
    tail = ranked.filter(pl.col("grouping_rank") > TAIL_RANK)
    hop_band_labels = [label for label, _, _ in DEFAULT_HOP_BANDS]
    grouping_band_labels = [label for label, _, _ in GROUPING_RANK_BANDS]

    return {
        "grouping rank": _paired_by_group(
            _mean_log2_rank(ranked, "grouping_band"), reference, "log2_rank", "grouping_band", grouping_band_labels
        ),
        "hop band": _paired_by_group(
            _mean_log2_rank(tail, "hop_band"), reference, "log2_rank", "hop_band", hop_band_labels
        ),
        "zone class": _paired_by_group(
            _mean_log2_rank(tail, "zone_class"), reference, "log2_rank", "zone_class", ZONE_CLASSES
        ),
    }


# =========================================
# Reads from the restored models
# =========================================


class GATProbe:
    """Records, on each forward pass of a ``GAT``, the node embeddings after its GATv2 layers and the input of each.

    Args:
        network: The GAT to probe.
    """

    def __init__(self, network: GAT):
        self.convs = list(network.convs.convs)
        self.embeddings: Tensor | None = None
        self.layer_inputs: list[tuple[tuple, dict]] = []
        self._handles = [network.convs.register_forward_hook(self._keep_embeddings)]
        for conv in self.convs:
            self._handles.append(conv.register_forward_pre_hook(self._keep_layer_input, with_kwargs=True))

    def attention(self) -> list[tuple[Tensor, Tensor]]:
        """Return the attention of each GATv2 layer on the last forward pass.

        Returns:
            One ``(edge_index, alpha)`` per layer: the edges attended over, self loops included, ``[2, num_edges]``,
            and the attention weight of each edge, ``[num_edges]``, averaged over the heads.
        """
        layer_attention = []
        with torch.no_grad():
            recorded_inputs = list(self.layer_inputs)
            for conv, (args, kwargs) in zip(self.convs, recorded_inputs, strict=True):
                _, (edge_index, alpha) = conv(*args, **kwargs, return_attention_weights=True)
                layer_attention.append((edge_index, alpha.mean(dim=-1)))

        return layer_attention

    def remove(self) -> None:
        """Remove the hooks from the network."""
        for handle in self._handles:
            handle.remove()

    def _keep_embeddings(self, module: torch.nn.Module, args: tuple, output: Tensor) -> None:
        self.embeddings = output.detach()

    def _keep_layer_input(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        self.layer_inputs.append((args, kwargs))


def embedding_similarity(embeddings: Tensor, rail_edge_index: Tensor, is_station: np.ndarray) -> dict[str, float]:
    """Return the mean cosine similarity of the embeddings of station nodes, joined by a rail edge or not.

    Args:
        embeddings: Embedding of every node, ``[num_nodes, d]``, or of every node for every user,
            ``[num_users, num_nodes, d]``. The similarities of the users are averaged.
        rail_edge_index: Rail edges in both directions, ``[2, num_rail_edges]``.
        is_station: Whether each node has a rail edge, ``[num_nodes]``.

    Returns:
        ``joined`` and ``not_joined``, the mean similarity over the pairs of distinct station nodes joined by a rail
        edge and over the other pairs of distinct station nodes, and ``gap``, the first minus the second.
    """
    user_embeddings = embeddings if embeddings.dim() == 3 else embeddings.unsqueeze(0)
    station_idx = torch.from_numpy(np.flatnonzero(is_station))
    num_nodes = user_embeddings.shape[1]

    is_joined = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    is_joined[rail_edge_index[0].cpu(), rail_edge_index[1].cpu()] = True
    station_joined = is_joined[station_idx][:, station_idx]
    is_distinct = ~torch.eye(len(station_idx), dtype=torch.bool)

    station_embeddings = torch.nn.functional.normalize(user_embeddings[:, station_idx].float().cpu(), dim=-1)
    similarity = (station_embeddings @ station_embeddings.transpose(1, 2)).mean(dim=0)
    joined = float(similarity[station_joined & is_distinct].mean())
    not_joined = float(similarity[~station_joined & is_distinct].mean())

    return {"joined": joined, "not_joined": not_joined, "gap": joined - not_joined}


def attention_summary(
    edge_index: Tensor, alpha: Tensor, num_nodes: int, classes: np.ndarray, contiguity_edge_index: Tensor
) -> pl.DataFrame:
    """Return the attention of each zone class over its neighbours, for one GATv2 layer.

    Args:
        edge_index: Edges attended over, self loops included, possibly over several graph copies of ``num_nodes``
            nodes each, ``[2, num_edges]``. The attention of an edge is the weight of its source in the update of its
            target.
        alpha: Attention weight of each edge, ``[num_edges]``.
        num_nodes: Number of nodes of one graph copy.
        classes: Class of every node, from ``zone_classes``.
        contiguity_edge_index: Contiguity edges of one copy, ``[2, num_contiguity_edges]``.

    Returns:
        Frame of ``zone_class, n_neighbours, norm_entropy, self_share, contiguity_share, rail_share``, one row per class.
        ``n_neighbours`` is the mean number of nodes attended over, self included. ``norm_entropy`` is the mean
        entropy of the attention of a target, divided by the log of its number of neighbours, over the targets with
        more than one. The shares split the attention of a target between itself, its contiguity neighbours and its
        other neighbours, the rail ones.
    """
    source_node = (edge_index[0] % num_nodes).cpu()
    target = edge_index[1].cpu()
    target_node = target % num_nodes
    alpha = alpha.detach().float().cpu()

    contiguity_keys = (contiguity_edge_index[0] * num_nodes + contiguity_edge_index[1]).cpu()
    is_self = source_node == target_node
    is_contiguity = torch.isin(source_node * num_nodes + target_node, contiguity_keys) & ~is_self
    is_rail = ~is_self & ~is_contiguity

    edges = pl.DataFrame({
        "target": target.numpy(),
        "zone_class": classes[target_node.numpy()].astype(str),
        "alpha": alpha.numpy(),
        "is_self": is_self.numpy(),
        "is_contiguity": is_contiguity.numpy(),
        "is_rail": is_rail.numpy(),
    })
    targets = edges.group_by("target", "zone_class").agg(
        n_neighbours=pl.len(),
        entropy=-(pl.col("alpha") * pl.col("alpha").clip(lower_bound=1e-12).log()).sum(),
        self_share=pl.col("alpha").filter(pl.col("is_self")).sum(),
        contiguity_share=pl.col("alpha").filter(pl.col("is_contiguity")).sum(),
        rail_share=pl.col("alpha").filter(pl.col("is_rail")).sum(),
    )
    norm_entropy = pl.when(pl.col("n_neighbours") > 1).then(pl.col("entropy") / pl.col("n_neighbours").log())

    return (
        targets
        .with_columns(norm_entropy=norm_entropy)
        .group_by("zone_class")
        .agg(pl.col("n_neighbours", "norm_entropy", "self_share", "contiguity_share", "rail_share").mean())
        .sort(pl.col("zone_class").replace_strict(ZONE_CLASSES, list(range(len(ZONE_CLASSES)))))
    )


def load_datamodule(run_path: Path) -> tuple[DictConfig, SurveyDataModule]:
    """Return the config of the run at ``run_path`` and its datamodule, set up on the split used by the run."""
    cfg = cast(DictConfig, OmegaConf.load(run_path / CONFIG_FILE))
    split = HouseholdSplit(val_size=cfg.train.val_size, test_size=cfg.train.test_size)
    datamodule = SurveyDataModule(
        cast(Config, cfg),
        split,
        seed=cfg.train.split_seed,
        batch_size=cfg.train.loader.batch_size,
        num_workers=0,
        project_root=PROJECT_ROOT,
    )
    datamodule.prepare_data()
    datamodule.setup()

    return cfg, datamodule


def restore_scorer(run_path: Path, cfg: DictConfig, name: str, seed: int, datamodule: SurveyDataModule) -> Scorer:
    """Return the scorer of the variant ``name`` at its best checkpoint for training ``seed``, in eval mode on the CPU.

    Raises:
        KeyError: The run has no variant named ``name``.
    """
    variants = {variant.name: variant for variant in build_variants(cfg.experiment.variants, dict(cfg.models))}
    if name not in variants:
        raise KeyError(f"The run has no variant `{name}`; its variants are {sorted(variants)}.")

    variant = variants[name]
    locations = datamodule.locations
    schema: DatasetSchema = datamodule.schema
    scorer = build_scorer(variant.model, variant.features, locations, schema, cfg.train.home_pe)

    checkpoint_path = run_path / CHECKPOINTS_DIR / f"{name}-seed{seed}" / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scorer_weights = checkpoint["state_dict"].items()
    scorer_state = {key.removeprefix("scorer."): value for key, value in scorer_weights if key.startswith("scorer.")}
    scorer.load_state_dict(scorer_state)

    return scorer.eval()


def first_test_users(datamodule: SurveyDataModule, num_users: int) -> UserBatch:
    """Return the first ``num_users`` users of the test split, joined with the locations."""
    rows: UserRows = next(iter(datamodule.test_dataloader()))
    first_rows = UserRows(*(field[:num_users] for field in rows))

    return UserBatch.join(first_rows, datamodule.locations)


def probe_scorer(scorer: Scorer, locations: Locations, batch: UserBatch) -> GATProbe:
    """Run ``scorer`` on ``batch`` with its GAT probed, and return the probe.

    Raises:
        TypeError: ``scorer`` is neither a ``GraphScorer`` over a ``GAT`` nor a ``TwoTowerScorer``.
    """
    if isinstance(scorer, TwoTowerScorer) and isinstance(scorer.node_tower.network, GAT):
        network = scorer.node_tower.network
    elif isinstance(scorer, GraphScorer) and isinstance(scorer.network, GAT):
        network = scorer.network
    else:
        raise TypeError(f"No GAT to probe in a {type(scorer).__name__}.")

    probe = GATProbe(network)
    with torch.no_grad():
        scorer(locations, batch)
    probe.remove()

    return probe


# =========================================
# Report
# =========================================


def print_edge_reads(data: str, run: int) -> None:
    """Print every detection read of run ``run`` of the dataset ``data``."""
    run_path = _run_path(data, run)
    cfg, datamodule = load_datamodule(run_path)
    locations = datamodule.locations
    num_nodes = locations.num_nodes
    rail_edges = locations.edges[EdgeKind.RAIL]
    contiguity_edges = locations.edges[EdgeKind.CONTIGUITY]

    degree = rail_degree(rail_edges.index, num_nodes)
    classes = zone_classes(degree, HUB_SHARE)
    is_station = degree > 0
    hub_idx = np.flatnonzero(classes == ZONE_CLASSES[0])
    hub_ids = locations.node_ids["loc_id"].to_numpy()[hub_idx]

    aggregate, per_user = read_report_run(run_path, cfg.train.max_recall_k)
    positives = positive_frame(per_user, classes)
    names = per_user["name"].unique().to_list()
    families = arm_families(names)
    mlp_name = grouping_model(names)
    ranks = grouping_ranks(positives, mlp_name)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(f"\n=== {data} run {run}: step 6 detection reads ===")
        print(
            f"station zones {int(is_station.sum())}, hubs {len(hub_idx)}: "
            + ", ".join(f"{loc_id} ({degree[node]})" for loc_id, node in zip(hub_ids, hub_idx, strict=True))
        )

        print("\n-- test pair accuracy, mean +- sd over seeds --")
        print(aggregate_metrics(aggregate, "test_pair_accuracy"))

        for family in families:
            family_positives = positives.filter(pl.col("name").is_in([family + suffix for suffix in ARM_SUFFIXES]))
            for reference_suffix in REFERENCE_SUFFIXES:
                reference = family + reference_suffix
                print(f"\n-- paired per-user avg_recall by zone class of each positive, vs {reference} --")
                print(paired_zone_split(family_positives, reference))

                for split_name, comparison in paired_rank_tail(family_positives, reference, ranks).items():
                    is_every_positive = split_name == "grouping rank"
                    scope = (
                        "every positive" if is_every_positive else f"positives beyond rank {TAIL_RANK} of {mlp_name}"
                    )
                    print(f"\n-- paired mean log2 rank by {split_name} ({mlp_name}), {scope}, vs {reference} --")
                    print(comparison)

        batch = first_test_users(datamodule, N_EMBEDDING_USERS)
        similarity_rows = []
        attention_rows = []
        for family in families:
            for suffix in ARM_SUFFIXES:
                name = family + suffix
                for seed in SEEDS:
                    scorer = restore_scorer(run_path, cfg, name, seed, datamodule)
                    probe = probe_scorer(scorer, locations, batch)
                    embeddings = probe.embeddings
                    if embeddings is None:
                        raise RuntimeError(f"The probe of `{name}` recorded no embeddings.")
                    if isinstance(scorer, GraphScorer):
                        embeddings = embeddings.reshape(batch.num_users, num_nodes, -1)

                    similarity = embedding_similarity(embeddings, rail_edges.index, is_station)
                    similarity_rows.append({"name": name, "seed": seed, **similarity})

                    for layer, (edge_index, alpha) in enumerate(probe.attention(), start=1):
                        summary = attention_summary(edge_index, alpha, num_nodes, classes, contiguity_edges.index)
                        attention_rows.append(
                            summary.with_columns(name=pl.lit(name), seed=pl.lit(seed), layer=pl.lit(layer))
                        )

        print(
            f"\n-- cosine similarity of station-zone embeddings, mean over seeds ({N_EMBEDDING_USERS} test users for the GAT) --"
        )
        print(
            pl
            .DataFrame(similarity_rows)
            .group_by("name")
            .agg(pl.col("joined", "not_joined", "gap").mean())
            .sort("name")
        )

        print("\n-- attention by zone class and layer, mean over seeds --")
        attention = pl.concat(attention_rows)
        print(
            attention
            .group_by("name", "layer", "zone_class")
            .agg(pl.col("n_neighbours", "norm_entropy", "self_share", "contiguity_share", "rail_share").mean())
            .sort("name", "layer", pl.col("zone_class").replace_strict(ZONE_CLASSES, list(range(len(ZONE_CLASSES)))))
        )


def _hop_band(hops: pl.Expr) -> pl.Expr:
    """Return the label of the ``DEFAULT_HOP_BANDS`` band each hop distance falls in."""
    return pl.coalesce([
        pl.when(hops.is_between(low, high)).then(pl.lit(label)) for label, low, high in DEFAULT_HOP_BANDS
    ])


def _rank_band(rank: pl.Expr) -> pl.Expr:
    """Return the label of the ``GROUPING_RANK_BANDS`` band each rank falls in, upper edge included."""
    return pl.coalesce([
        pl.when((rank > low) & (rank <= high)).then(pl.lit(label)) for label, low, high in GROUPING_RANK_BANDS
    ])


def _mean_log2_rank(positives: pl.DataFrame, group: str) -> pl.DataFrame:
    """Return the mean ``log2_rank`` of the positives of every model, seed, user and ``group``."""
    return positives.group_by("name", "seed", "user_idx", group).agg(pl.col("log2_rank").mean())


def _paired_by_group(
    values: pl.DataFrame, reference: str, value: str, group: str, group_order: Sequence[str]
) -> pl.DataFrame:
    """Return ``paired_comparison`` of the per-user ``value`` against ``reference``, run on each group in order."""
    comparisons = []
    stage_values = values.with_columns(stage=pl.lit(PER_USER_STAGE))
    for label in group_order:
        group_values = stage_values.filter(pl.col(group) == label)
        if group_values.filter(pl.col("name") == reference).is_empty():
            continue
        comparison = paired_comparison(group_values, reference, value, n_bootstrap=N_BOOTSTRAP)
        comparisons.append(comparison.with_columns(group=pl.lit(label)))

    return pl.concat(comparisons).select("group", pl.all().exclude("group"))


def _run_path(data: str, run: int) -> Path:
    """Return the directory of run ``run`` of the dataset ``data``, under the default runs path."""
    cfg = OmegaConf.load(PROJECT_ROOT / "activitygraphs" / "conf" / "data" / f"{data}.yaml")
    return run_dir(PROJECT_ROOT / "reports" / "runs", cfg.name, run)


if __name__ == "__main__":
    print_edge_reads(sys.argv[1], int(sys.argv[2]))
