"""Diagnostic metrics: contiguity-hop distance and distance-from-home hop-band ranking metrics.

Aggregate ranking metrics are dominated by near-home positives. To detect whether a model
loses predictive power beyond its message-passing receptive field, ``ActivityGraphModule``
groups each node by its spatial adjacency-hop distance from the user's home node into hop bands.
It reports recall@k / ndcg@k within each band, using the per-hop-band torchmetrics collections
built here. Regular per-step ranking metrics use ``torchmetrics`` directly in the Lightning module.
"""

from collections.abc import Collection

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torchmetrics import MeanMetric, Metric, MetricCollection, SumMetric
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall
from torchmetrics.utilities.data import dim_zero_cat

# Hop bands: (label, low_hops_inclusive, high_hops_inclusive)
HopBands = Collection[tuple[str, float, float]]

DEFAULT_HOP_BANDS: HopBands = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13+", 13, float("inf")),
)


def compute_home_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return the all-pairs contiguity-hop distance matrix ``[num_nodes, num_nodes]``.

    Treats the graph as undirected and unweighted; unreachable pairs are ``inf``.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))

    return shortest_path(adjacency, method="D", unweighted=True, directed=False)


class HopBandMetrics(torch.nn.Module):
    """Ranking, NLL and positive-count metrics computed separately within each hop band.

    Ranking is restricted to the nodes inside a band, so each band answers "within this distance
    ring, does the model separate visited from unvisited nodes" - the distance decay a model gets
    for free from home is factored out. ``compute`` returns one row per band in long format, which
    both the scalar logging keys and the W&B ``hop_bands`` table are derived from.
    """

    def __init__(self, hop_bands: HopBands, k: int) -> None:
        super().__init__()
        self.hop_bands = tuple(hop_bands)
        self.k = k
        self.ranking = torch.nn.ModuleDict({
            label: MetricCollection({
                "recall": RetrievalRecall(top_k=k, empty_target_action="skip"),
                "ndcg": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            })
            for label, _, _ in self.hop_bands
        })
        self.nll = torch.nn.ModuleDict({label: MeanMetric() for label, _, _ in self.hop_bands})
        self.n_pos = torch.nn.ModuleDict({label: SumMetric() for label, _, _ in self.hop_bands})
        self._seen: set[str] = set()

    def update(self, hops: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, user_ids: torch.Tensor) -> None:
        """Accumulate one batch, given each node's hop distance from its user's home node."""
        scores = logits.sigmoid()

        for label, low, high in self.hop_bands:
            mask = (hops >= low) & (hops <= high)
            if not mask.any():
                continue

            band_target = target[mask]
            self.ranking[label].update(scores[mask], band_target, indexes=user_ids[mask])
            self.nll[label].update(
                F.binary_cross_entropy_with_logits(logits[mask], band_target.float(), reduction="none")
            )
            self.n_pos[label].update(band_target.sum())

            if band_target.sum() > 0:
                self._seen.add(label)

    def compute(self) -> list[dict[str, float | str]]:
        """Return one row per band, in band order, skipping bands that saw no positives.

        Bands with no positives are skipped. The lower bound of the hop band, ``hop_low``, is
        included as a numeric sort key.
        """
        rows: list[dict[str, float | str]] = []

        for label, low, _ in self.hop_bands:
            if label not in self._seen:
                continue

            row: dict[str, float | str] = {"hop_band": label, "hop_low": low, "k": self.k}
            row |= {name: value.item() for name, value in self.ranking[label].compute().items()}
            row |= {"nll": self.nll[label].compute().item(), "n_pos": self.n_pos[label].compute().item()}
            rows.append(row)

        return rows

    def reset(self) -> None:
        for label, _, _ in self.hop_bands:
            self.ranking[label].reset()
            self.nll[label].reset()
            self.n_pos[label].reset()
        self._seen.clear()


def hop_band_scalars(rows: Collection[dict[str, float | str]]) -> dict[str, float]:
    """Flatten hop-band rows into ``test_hop_<label>_<metric>`` scalar logging keys.

    Ranking metrics keep their ``@k`` suffix (e.g. ``test_hop_3-5_recall@25``); ``nll`` and
    ``n_pos`` do not, since they are not cutoff-dependent.
    """
    scalars = {}

    for row in rows:
        prefix = f"test_hop_{row['hop_band']}_"
        for name in ("recall", "ndcg"):
            scalars[f"{prefix}{name}@{row['k']}"] = float(row[name])
        for name in ("nll", "n_pos"):
            scalars[f"{prefix}{name}"] = float(row[name])

    return scalars


class RetrievalRPrecision(Metric):
    """Per-user recall at k = number of visited nodes (R-precision).

    For each user, scores the top-R nodes where R is that user's realised set size |RG_i|,
    and reports hits / R, averaged over users. Users that have no visits are dropped from the average (e.g. when only
    home node is visited and is dropped, masked out by the caller). ``n_dropped_users`` tracks the number of such users.

    R-precision is comparable across dataset since it compares over the realised set size.
    """

    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_dropped", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        for idx in torch.unique(indexes):
            mask = indexes == idx
            t = target[mask]
            r = int(t.sum())
            if r == 0:
                self.n_dropped = self.n_dropped + 1
                continue
            top = preds[mask].topk(min(r, t.numel())).indices
            self.score_sum = self.score_sum + t[top].sum() / r
            self.n_users = self.n_users + 1

    def compute(self) -> torch.Tensor:
        return self.score_sum / self.n_users.clamp(min=1)

    def user_counts(self) -> dict[str, float]:
        """Returns a dictionary with two items: the count of users that were scored and the count of users that were
        dropped (due to RG_i too small)."""
        return {"n_scored_users": float(self.n_users), "n_dropped_users": float(self.n_dropped)}


class PerUserRanking(Metric):
    """Ranking scores for each test user, disaggregate.

    Keeps the ranking scores and positive counts of each test user for futher analyses, as a struct of lists, indexed
    by users:
        - n_pos: number of positives for that user
        - n_pos_home_incl: number of positives for that user, including the home node
        - r_precision: R-precision for this user
        - recall: recall@k for this user

    Also keeps info about each of the users' positives (i.e. visited nodes, R = |RG_i|), NOT indexed by user, but by
    ``sum(num_pos_u, u in users)`` (i.e. each index is a visited node), struct of flattened lists:
        - pos_hops: number of hops from home to the positive node
        - pos_in_top_r: did this positive node make it into the top R (= |RG_i|) nodes, ties broken arbitrarily
        - pos_in_top_k: did this positive node make it into the top k nodes, ties broken arbitrarily
        - pos_n_scored_higher: number of nodes ranked higher than the positive node
        - pos_n_tied: number of nodes that tied with the positive node
    """

    full_state_update = False

    user_id: list[torch.Tensor]
    n_pos: list[torch.Tensor]
    n_pos_home_incl: list[torch.Tensor]
    r_precision: list[torch.Tensor]
    recall: list[torch.Tensor]

    pos_hops: list[torch.Tensor]
    pos_in_top_r: list[torch.Tensor]
    pos_in_top_k: list[torch.Tensor]
    pos_n_scored_higher: list[torch.Tensor]
    pos_n_tied: list[torch.Tensor]

    score_vector: list[torch.Tensor]

    _FIELDS = ("user_id", "n_pos", "n_pos_home_incl", "r_precision", "recall")
    _POSITIVE_FIELDS = ("pos_hops", "pos_in_top_r", "pos_in_top_k", "pos_n_scored_higher", "pos_n_tied")

    def __init__(self, k: int, store_score_vectors: bool = False) -> None:
        super().__init__()
        self.k = k
        self.store_score_vectors = store_score_vectors
        for field in self._FIELDS + self._POSITIVE_FIELDS:
            self.add_state(field, default=[], dist_reduce_fx="cat")

        self.add_state("score_vector", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        exclude: torch.Tensor,
        hops: torch.Tensor,
    ) -> None:
        """Update the per-user and per-positive tensors with the results from the current batch. Skips users with no
        positives.

        Args:
            preds: flat tensor of predicted scores, [n_users * n_nodes]
            target: tensor of binary visit labels, aligned with ``preds``, [n_users * n_nodes]
            indexes: tensor of user ids of each row, [n_users * n_nodes]
            exclude: boolean mask of the rows to be excluded from ranking (e.g. ``is_home`` flag), , aligned with
                ``preds``, [n_users * n_nodes]
            hops: tensor of hop distances for each node, aligned with ``preds``, [n_users * n_nodes]
        """
        for idx in torch.unique(indexes):
            mask = indexes == idx

            full_target = target[mask]
            full_scores = preds[mask]

            non_excluded_nodes = ~exclude[mask]
            scored_target = full_target[non_excluded_nodes]
            scored_scores = full_scores[non_excluded_nodes]
            scored_hops = hops[mask][non_excluded_nodes]

            num_positives = int(scored_target.sum())
            if num_positives == 0:
                continue

            top_r = scored_scores.topk(min(num_positives, scored_target.numel())).indices
            top_k = scored_scores.topk(min(self.k, scored_target.numel())).indices

            full_num_positives = full_target.sum()
            r_precision = scored_target[top_r].sum() / num_positives
            recall = scored_target[top_k].sum() / num_positives

            is_positive = scored_target.bool()
            in_top_r = torch.zeros_like(is_positive)
            in_top_r[top_r] = True
            in_top_k = torch.zeros_like(is_positive)
            in_top_k[top_k] = True

            positive_scores = scored_scores[is_positive].unsqueeze(1)
            n_scored_higher = (scored_scores.unsqueeze(0) > positive_scores).sum(dim=1)
            n_tied = (scored_scores.unsqueeze(0) == positive_scores).sum(dim=1) - 1

            self.user_id.append(idx.reshape(1).long())
            self.n_pos.append(torch.full((1,), num_positives, dtype=torch.long, device=scored_scores.device))
            self.n_pos_home_incl.append(full_num_positives.reshape(1).long())
            self.r_precision.append(r_precision.reshape(1).float())
            self.recall.append(recall.reshape(1).float())

            self.pos_hops.append(scored_hops[is_positive].float())
            self.pos_in_top_r.append(in_top_r[is_positive])
            self.pos_in_top_k.append(in_top_k[is_positive])
            self.pos_n_scored_higher.append(n_scored_higher.long())
            self.pos_n_tied.append(n_tied.long())

            if self.store_score_vectors:
                self.score_vector.append(full_scores.detach().float().reshape(1, -1).cpu())

    def columns(self) -> dict[str, torch.Tensor]:
        """Return the per-user columns as a mapping (col_name -> tensor of values), values are indexed by users.  See
        class docstring for description of columns.

        Use this rather than ``compute``: the torchmetrics wrapper around ``compute`` squeezes
        single-element outputs to 0-dim, which would turn a one-user split into a scalar.
        """
        return {field: dim_zero_cat(getattr(self, field)) for field in self._FIELDS}

    def positive_columns(self) -> dict[str, list[list]]:
        """Return the positive (visited nodes) columns as a mapping (col_name -> tensor of values), values are indexed
        by positives (i.e. ``sum(num_pos_u for u in users)``). See class docstring for description of columns.
        """
        values = {pos_field: dim_zero_cat(getattr(self, pos_field)).tolist() for pos_field in self._POSITIVE_FIELDS}
        positives: dict[str, list[list]] = {pos_field: [] for pos_field in self._POSITIVE_FIELDS}

        start = 0
        for count in dim_zero_cat(self.n_pos).tolist():
            end = start + count
            for pos_field in self._POSITIVE_FIELDS:
                positives[pos_field].append(values[pos_field][start:end])
            start = end

        return positives

    def score_vectors(self) -> torch.Tensor:
        """Return the tensor of model score vectors, row-aligned with ``columns()``, ``[n_users, num_nodes]``. Empty
        when ``store_score_vectors`` is off."""
        if not self.score_vector:
            return torch.empty(0)

        return dim_zero_cat(self.score_vector)

    def compute(self) -> dict[str, torch.Tensor]:
        return self.columns()
