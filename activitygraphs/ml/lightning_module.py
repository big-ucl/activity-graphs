"""ActivityGraphModule and _EpochMetricsCallback for Lightning-based GNN training."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalPrecision, RetrievalRecall

from activitygraphs.ml.dataset import ActivityDataset, is_home_node_mask
from activitygraphs.ml.losses import BCELoss, Loss
from activitygraphs.ml.metrics import (
    DEFAULT_HOP_BANDS,
    HopBandMetrics,
    PerUserRanking,
    RetrievalRPrecision,
    hop_band_scalars,
)
from activitygraphs.ml.sampling import pps_sampling

HOME_PE_BINS = 8

# Sampling weight given to a node outside the scored candidate set: its sigmoid is zero, so PPS
# never draws it while any scored candidate is left.
EXCLUDED_LOGIT = -1e9


def extract_features(
    batch: pyg.data.Batch,
    full_info: bool,
    is_home_idx: int,
    use_demographics: bool = True,
    pop_logit: torch.Tensor | None = None,
    home_hop_distance: torch.Tensor | None = None,
    home_pe_bins: int = HOME_PE_BINS,
) -> torch.Tensor:
    """Return node features for a batch.

    Starts with ``batch.x`` (which already contains the network features and the per-user
    ``is_home`` indicator).

     If ``pop_logit`` is provided, the ``NodeBaseline`` popularity logits for each node are appended
    (``1`` column) is appended.

     If ``home_hop_distance`` is provided, a home-anchored positional encoding is added (RBF expansion of
     hop-distance-to-home,``n_bins`` columns) is appended. Uses ``is_home_idx`` as the column for the home index.

    If ``use_demographics`` is True, the per-graph user-specific demographics (``batch.graph_x``) are broadcast over
    nodes and concatenated.

    If ``full_info`` is True, the per-node distance-from-home (``batch.distances``) is appended as the last column;
    this is used only by the distance-augmented MLP baseline so that the explicit distance-to-home signal does not leak
    into the other models.
    """
    x = cast(torch.Tensor, batch.x)

    if pop_logit is not None:
        pop_logit_col = create_pop_logit_column(pop_logit, batch, standardize=True)
        x = torch.cat([x, pop_logit_col], dim=-1)

    if home_hop_distance is not None:
        home_pe = create_home_distance_encoding(batch, home_hop_distance, is_home_idx, n_bins=home_pe_bins)
        x = torch.cat([x, home_pe], dim=-1)

    if use_demographics:
        x = torch.cat([x, batch.graph_x[batch.batch]], dim=-1)

    if not full_info:
        return x

    return torch.cat([x, batch.distances.to(x.dtype)], dim=-1)


def extracted_features_dim(
    dataset: ActivityDataset,
    use_demographics: bool = True,
    full_info: bool = False,
    use_pop_feature: bool = False,
    use_home_pe: bool = False,
    n_bins: int = HOME_PE_BINS,
) -> int:
    """Output width of extract_features for this dataset, given the same flags."""
    dim = dataset.num_features

    if use_pop_feature:
        dim += 1
    if use_home_pe:
        dim += n_bins
    if use_demographics:
        dim += dataset.demographics.shape[1]
    if full_info:
        dim += dataset.distances.shape[-1]

    return dim


def create_pop_logit_column(pop_logit: torch.Tensor, batch: pyg.data.Batch, standardize: bool):
    """Per-node popularity logit aligned to the nodes in `batch`, shape [num_nodes, 1].

    `standardize` z-scores it over the node vector (for the pop-logit-as-input-feature variant).
    """
    pop_logit_col = pop_logit[_compute_node_index_within_graph(batch, pop_logit.device)]
    if standardize:
        pop_logit_col = (pop_logit_col - pop_logit.mean()) / pop_logit.std().clamp(min=1e-6)

    return pop_logit_col.unsqueeze(-1)


def _compute_node_index_within_graph(batch: pyg.data.Batch, device: torch.device) -> torch.Tensor:
    """Index of each node within its own graph (0..N-1), aligned to batch.x row order."""
    return torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]


def compute_home_hops(batch: pyg.data.Batch, home_hop_distance: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Hop distance from each node to the user's home node in the graph, shape ``[num_nodes]``."""
    device = batch.x.device
    node_idx = _compute_node_index_within_graph(batch, device)
    is_home = is_home_node_mask(batch.x, is_home_idx)

    home = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
    home[batch.batch[is_home]] = node_idx[is_home]

    return home_hop_distance[home[batch.batch], node_idx]


def create_home_distance_encoding(
    batch: pyg.data.Batch, home_hop_distance: torch.Tensor, is_home_idx: int, n_bins: int = 8
) -> torch.Tensor:
    """Positional-encoding based on distance from user home (single anchor). RBF expansion of num. hops to home for
    each node, shape ``[num_nodes, n_bins]``."""

    hops = compute_home_hops(batch, home_hop_distance, is_home_idx).clamp(max=40)
    centers = torch.linspace(0, 40, n_bins, device=batch.x.device)

    return torch.exp(-((hops.unsqueeze(-1) - centers) ** 2) / 2.0)  # [num_nodes, n_bins]


@dataclass(frozen=True)
class RankingTensors:
    """Set of tensors read by the ranking metrics (R-Precision, Recall@K, etc). The tensors are restricted to the
    non-excluded/scored candidates (i.e. home node is excluded)."""

    logits: Tensor
    probs: Tensor
    target: Tensor
    users: Tensor
    scored: Tensor

    def restrict(self, values: Tensor) -> Tensor:
        """Restrict a tensor to the non-excluded scored nodes."""
        return values[self.scored]


@dataclass(frozen=True)
class EvaluationTensors:
    """Set of tensors computed during the validation and testing steps. Metrics are computed using these tensors.
    All nodes are included in the tensors, whereas the ``ranking`` field contains the set of tensors with the home node
    excluded (used for the ranking metrics)."""

    logits: Tensor
    loss: Tensor
    bce: Tensor
    bce_weighted: Tensor
    probs: Tensor
    target: Tensor
    users: Tensor
    ranking: RankingTensors


class ActivityGraphModule(L.LightningModule):
    """LightningModule wrapping any GNN model for node-level binary prediction on activity graphs.

    Handles weighted BCE training, optional L1 regularisation, and per-epoch ranking metrics.
    Models must implement ``forward(x, edge_index, edge_attr, batch) -> logits``.

    Args:
        model: Any ``nn.Module`` with the GNN forward signature.
        lr: Initial learning rate for AdamW.
        pos_weight: Scalar positive-class weight for BCE loss, computed from the train split.
        is_home_idx: Column index of ``is_home`` in the node feature matrix.
        loss: ``Loss`` instance. Defaults to BCE if none.
        reg: Optional regularisation type; only ``"l1"`` is supported.
        lambda_reg: L1 coefficient (ignored when ``reg`` is None).
        full_info: If True, augment node features with home indicator and distances.
        use_demographics: concatenate the per-user demographics onto the node features.
        k: Rank cutoff for precision, recall, and NDCG metrics.
        weight_decay: AdamW weight decay.
        schedule_lr: add a ReduceLROnPlateau scheduler to the optimizer, defaults to False.
        home_hop_distance: ``[num_nodes, num_nodes]`` contiguity-hop distance matrix enabling
            distance-from-home hop-band ranking metrics at test time.
        hop_bands: Hop-distance bands for the hop-band metrics.
        pop_logit: Logits of global per-node visit frequencies.
        pop_mode: "none"=do not inject ``pop_logits``; "offset"=inject in the loss function, "feature"=inject as
            features to the model.
        use_home_pe: use home-anchored positional encodings in features, default False.
        home_pe_bins: number of bins for the RBF expansion of the home PEs, default ``HOME_PE_BINS``.
        compile_model: torch.compile the inner NN, default False.
        store_score_vectors: retain each test user's full per-node score vector.
        log_train_ranking: log ``train_r_precision`` on the training batches, default True. Turn it off when per-user
            computation becomes expensive.
    """

    pos_weight: torch.Tensor  # registered buffer; annotated so it types as Tensor, not Tensor | Module
    home_hop_distance: torch.Tensor  # registered buffer, same reason

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        pos_weight: torch.Tensor,
        is_home_idx: int,
        home_hop_distance: np.ndarray | torch.Tensor,
        loss: Loss | None = None,
        reg: str | None = None,
        lambda_reg: float = 0.01,
        full_info: bool = False,
        use_demographics: bool = True,
        k: int = 5,
        weight_decay: float = 1e-4,
        schedule_lr: bool = False,
        hop_bands: Collection[tuple[str, float, float]] = DEFAULT_HOP_BANDS,
        pop_logit: torch.Tensor | None = None,
        pop_mode: Literal["none", "offset", "feature"] = "none",
        use_home_pe: bool = False,
        home_pe_bins: int = HOME_PE_BINS,
        compile_model: bool = False,
        store_score_vectors: bool = False,
        log_train_ranking: bool = True,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = loss if loss is not None else BCELoss
        self.register_buffer("pos_weight", pos_weight)
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.full_info = full_info
        self.use_demographics = use_demographics
        self.k = k
        self.weight_decay = weight_decay
        self.schedule_lr = schedule_lr

        # Metrics: General setup
        metrics = MetricCollection({
            "r_precision": RetrievalRPrecision(),
            f"recall@{k}": RetrievalRecall(top_k=k, empty_target_action="skip"),
            f"ndcg@{k}": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            f"precision@{k}": RetrievalPrecision(top_k=k, empty_target_action="skip"),
        })

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.log_train_ranking = log_train_ranking
        self.train_r_precision = RetrievalRPrecision() if log_train_ranking else None

        self.val_r_precision_home_incl = RetrievalRPrecision()  # With home included metrics kept as sanity checks
        self.test_r_precision_home_incl = RetrievalRPrecision()

        self.val_calibration = BinaryCalibrationError(n_bins=15, norm="l1")
        self.test_calibration = BinaryCalibrationError(n_bins=15, norm="l1")

        # Metrics: distance-from-home hop-band metrics (test time only).
        self.is_home_idx = is_home_idx
        self.register_buffer(
            "home_hop_distance", torch.as_tensor(home_hop_distance, dtype=torch.float), persistent=False
        )
        self.hop_band_metrics = HopBandMetrics(hop_bands, k)

        # Hop-band results in long format (one row per band), populated by `on_test_epoch_end`.
        self.hop_band_rows: list[dict[str, float | str]] = []

        # Metrics: per-user test scores retained for paired model comparison, in column format
        # (one entry per test user), populated by `on_test_epoch_end`.
        self.test_per_user = PerUserRanking(k, store_score_vectors=store_score_vectors)
        self.per_user_columns: dict[str, list[float]] = {}

        # Full per-node score vector of each test user, kept only when `store_score_vectors` is set.
        self.per_user_score_vectors: list[list[float]] = []

        # Metrics: Capture predicted and true expected |RG_i| sizes
        self.test_pred_size = MeanMetric()
        self.test_true_size = MeanMetric()

        # Metrics: evaluate actual Poisson-sampled sets
        self.test_sampled_recall = MeanMetric()
        self.test_sampled_size = MeanMetric()

        # Training: Population logits injection
        if pop_mode not in ("none", "offset", "feature"):
            raise ValueError(f"pop_mode must be none|offset|feature, got {pop_mode}")
        if pop_mode != "none" and pop_logit is None:
            raise ValueError(f"pop_mode must be none|offset, got {pop_mode}")

        self.pop_mode = pop_mode
        if pop_logit is not None:
            self.register_buffer("pop_logit", pop_logit, persistent=False)
        else:
            self.pop_logit = None

        # Training: Home-anchored positional encodings injection
        self.use_home_pe = use_home_pe
        self.home_pe_bins = home_pe_bins

        # Training: torch.compile (applied in configure_model, not here)
        self.compile_model = compile_model
        self._compiled = False

    def configure_model(self) -> None:
        """Compile the inner NN with ``torch.compile``."""
        if self._compiled or not self.compile_model:
            return
        if self.trainer.fast_dev_run or self.trainer.overfit_batches:
            return

        self.model = torch.compile(self.model, dynamic=True)  # dynamic=True since graph sizes (i.e. batch sizes) differ
        self._compiled = True

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, batch: pyg.data.Batch, batch_idx: int) -> torch.Tensor:
        out = self.compute_logits(batch)
        loss = self.loss.loss_fn(out, batch)

        if self.reg == "l1":
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + self.lambda_reg * l1_norm

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        if self.train_r_precision is not None:
            ranking_tensors = self._compute_ranking_tensors(batch, out)
            self.train_r_precision.update(ranking_tensors.probs, ranking_tensors.target, indexes=ranking_tensors.users)
            self.log(
                "train_r_precision",
                self.train_r_precision,
                on_step=False,
                on_epoch=True,
                batch_size=batch.num_nodes,
            )

        return loss

    def _compute_ranking_tensors(self, batch: pyg.data.Batch, logits: Tensor) -> RankingTensors:
        scored_mask = ~is_home_node_mask(batch.x, self.is_home_idx)
        flat_logits = logits.squeeze(-1)

        return RankingTensors(
            logits=flat_logits[scored_mask],
            probs=flat_logits.sigmoid()[scored_mask],
            target=batch.y.squeeze(-1).long()[scored_mask],
            users=batch.user_id[batch.batch][scored_mask],
            scored=scored_mask,
        )

    def _compute_evaluation_tensors(self, batch) -> EvaluationTensors:
        out = self.compute_logits(batch)

        generator = torch.Generator(device=out.device).manual_seed(42)
        loss = self.loss.loss_fn(out, batch, generator)
        bce = F.binary_cross_entropy_with_logits(out, batch.y.float())
        bce_weighted = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=self.pos_weight)
        probs = out.squeeze(-1).sigmoid()
        target = batch.y.squeeze(-1)
        users = batch.user_id[batch.batch]
        ranking_tensors = self._compute_ranking_tensors(batch, out)

        return EvaluationTensors(
            logits=out,
            loss=loss,
            bce=bce,
            bce_weighted=bce_weighted,
            probs=probs,
            target=target,
            users=users,
            ranking=ranking_tensors,
        )

    def _num_scored_and_dropped_users(self, stage: Literal["val", "test"]) -> dict[str, float]:
        """Returns a dictionary of the number of scored and dropped users for a given stage. Each key is
        ``{stage}_{name}``, with name being either "n_scored_users" or "n_dropped_users"."""
        collection = self.val_metrics if stage == "val" else self.test_metrics
        r_precision: RetrievalRPrecision = cast(RetrievalRPrecision, collection[f"{stage}_r_precision"])
        user_counts = r_precision.user_counts()

        return {f"{stage}_{name}": value for name, value in user_counts.items()}

    def validation_step(self, batch, batch_idx: int) -> None:
        eval_tensors = self._compute_evaluation_tensors(batch)
        ranking_tensors = eval_tensors.ranking

        # Log loss metrics
        log_loss_kwargs = {"on_step": False, "on_epoch": True, "batch_size": batch.num_nodes}
        if self.loss.name != "bce":
            self.log(f"val_{self.loss.name}", eval_tensors.loss, **log_loss_kwargs)
        self.log("val_bce", eval_tensors.bce, **log_loss_kwargs)
        self.log("val_bce_weighted", eval_tensors.bce_weighted, **log_loss_kwargs)

        # Ranking metrics
        # torchmetrics Retrieval* treat preds as probabilities and drop preds <= 0, so feed sigmoid
        # (monotonic, preserves ranking) rather than raw logits.
        self.val_metrics.update(ranking_tensors.probs, ranking_tensors.target, indexes=ranking_tensors.users)
        self.val_r_precision_home_incl.update(  # includes all nodes (home node included, sanity check)
            eval_tensors.probs, eval_tensors.target.long(), indexes=eval_tensors.users
        )

        # Calibration metrics
        self.val_calibration.update(eval_tensors.probs, eval_tensors.target.long())

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("val"))
        self.val_metrics.reset()

        self.log("val_r_precision_home_incl", self.val_r_precision_home_incl.compute())
        self.val_r_precision_home_incl.reset()

        self.log("val_calibration_l1", self.val_calibration.compute())
        self.val_calibration.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        eval_tensors = self._compute_evaluation_tensors(batch)
        ranking = eval_tensors.ranking

        # Log loss metrics
        log_loss_kwargs = {"on_step": False, "on_epoch": True, "batch_size": batch.num_nodes}
        if self.loss.name != "bce":
            self.log(f"test_{self.loss.name}", eval_tensors.loss, **log_loss_kwargs)
        self.log("test_bce", eval_tensors.bce, **log_loss_kwargs)
        self.log("test_bce_weighted", eval_tensors.bce_weighted, **log_loss_kwargs)

        # Ranking metrics
        self.test_metrics.update(  # Feed sigmoid to conform to torchmetrics calling convention (see validation step)
            ranking.probs, ranking.target, indexes=ranking.users
        )
        self.test_r_precision_home_incl.update(  # includes all nodes (home node included, sanity check)
            eval_tensors.probs, eval_tensors.target.long(), indexes=eval_tensors.users
        )

        # Per-user metrics
        self.test_per_user.update(  # All nodes included, drops home node itself instead to allow full per-user comparison
            eval_tensors.probs, eval_tensors.target.long(), indexes=eval_tensors.users, exclude=~ranking.scored
        )

        # Hop-band metrics
        hops = ranking.restrict(compute_home_hops(batch, self.home_hop_distance, self.is_home_idx))
        self.hop_band_metrics.update(hops, ranking.logits, ranking.target, ranking.users)

        # Calibration metrics
        self.test_calibration.update(eval_tensors.probs, eval_tensors.target.long())

        # True and predicted expected |RG_i| set sizes
        for idx in torch.unique(ranking.users):
            m = ranking.users == idx
            self.test_pred_size.update(ranking.probs[m].sum())
            self.test_true_size.update(ranking.target[m].sum())

        # Update sampled set recall and size metrics
        self._update_sampled_sets_metrics(batch, eval_tensors)

    def _update_sampled_sets_metrics(self, batch: pyg.data.Batch, step: EvaluationTensors, n_draws: int = 8):
        # The one ranking read that cannot take the narrow view: ``pps_sampling`` groups by
        # ``batch.batch``, so it needs a tensor per node. Excluded nodes are given a sampling weight
        # of zero instead of being removed, which leaves that grouping intact.
        excluded = ~step.ranking.scored
        logits = step.logits.squeeze(-1).masked_fill(excluded, EXCLUDED_LOGIT)
        target = step.target.float().masked_fill(excluded, 0.0)
        users = step.users
        gen = torch.Generator(device=logits.device).manual_seed(42)

        # pos_weight=None under plain-BCE training (the default); pass float(self.pos_weight) only
        # if the model was trained with weighted BCE.
        for _ in range(n_draws):
            chosen = pps_sampling(self.k, logits, batch.batch, gen).squeeze(-1)
            for idx in torch.unique(users):
                m = users == idx
                r = target[m].sum()
                if r == 0:
                    continue
                hits = (chosen[m] * target[m]).sum()
                self.test_sampled_recall.update(hits / r)
                self.test_sampled_size.update(chosen[m].sum())

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("test"))
        self.test_metrics.reset()

        # Not logged: these are per-user vectors, not scalars. Persisted by `run_experiment`.
        self.per_user_columns = {name: value.cpu().tolist() for name, value in self.test_per_user.columns().items()}
        self.per_user_score_vectors = self.test_per_user.score_vectors().cpu().tolist()
        self.test_per_user.reset()

        self.hop_band_rows = self.hop_band_metrics.compute()
        self.log_dict(hop_band_scalars(self.hop_band_rows))
        self.hop_band_metrics.reset()

        self.log("test_r_precision_home_incl", self.test_r_precision_home_incl.compute())
        self.test_r_precision_home_incl.reset()

        self.log("test_calibration_l1", self.test_calibration.compute())
        self.test_calibration.reset()

        self.log("test_pred_size", self.test_pred_size.compute())
        self.test_pred_size.reset()

        self.log("test_true_size", self.test_true_size.compute())
        self.test_true_size.reset()

        self.log("test_sampled_recall", self.test_sampled_recall.compute())
        self.test_sampled_recall.reset()

        self.log("test_sampled_size", self.test_sampled_size.compute())
        self.test_sampled_size.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Only add a scheduler if not trying to overfit (i.e. not in diagnostic mode). Validation error is not a useful
        # signal when purposefully overfitting
        if self.trainer.overfit_batches == 0 and self.schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.loss.monitor_mode, factor=0.5, patience=5
            )

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": self.loss.monitor}}

        return optimizer

    def compute_logits(self, batch: pyg.data.Batch) -> torch.Tensor:
        """Model logits for a batch, includes node-popularity feature column or offset if configured."""
        pop_logits = self.pop_logit if self.pop_mode == "feature" else None
        home_hop_distance = self.home_hop_distance if self.use_home_pe else None

        x = extract_features(
            batch,
            self.full_info,
            self.is_home_idx,
            use_demographics=self.use_demographics,
            pop_logit=pop_logits,
            home_hop_distance=home_hop_distance,
            home_pe_bins=self.home_pe_bins,
        )
        out = self(x, batch.edge_index, batch.edge_attr, batch.batch)

        if self.pop_mode == "offset":
            out = out + create_pop_logit_column(self.pop_logit, batch, standardize=False)

        return out
