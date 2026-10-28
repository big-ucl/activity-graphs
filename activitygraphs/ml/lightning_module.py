"""RankingModule and _EpochMetricsCallback for Lightning-based GNN training."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryCalibrationError

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.dataset import VisitDataset, is_home_node_mask
from activitygraphs.ml.losses import Loss
from activitygraphs.ml.metrics import PerUserRanking, RetrievalAverageRecall, RetrievalRPrecision, per_user_recall

HOME_PE_BINS = 8
HOME_PE_MAX_HOPS = 40
CALIBRATION_BINS = 15


def extract_features(
    batch: UserBatch,
    full_info: bool,
    is_home_idx: int,
    use_demographics: bool = True,
    home_hop_distance: torch.Tensor | None = None,
    home_pe_bins: int = HOME_PE_BINS,
    home_pe_max_hops: int = HOME_PE_MAX_HOPS,
    use_rwpe: bool = False,
) -> torch.Tensor:
    """Return node features for a batch.

    Starts with the location features ``batch.x_loc`` joined with the per-user node features ``batch.x_user_node``
    (which contain the ``is_home`` indicator).

    If ``use_rwpe`` is True, the random-walk positional encoding (``batch.random_walk_pe``) is appended.

    If ``home_hop_distance`` is provided, a home-anchored positional encoding is added (RBF expansion of
    hop-distance-to-home,``n_bins`` columns) is appended. Uses ``is_home_idx`` as the column of ``is_home`` in
     ``batch.x_user_node``.

    If ``use_demographics`` is True, the per-graph user attributes (``batch.user_attr``) are broadcast over
    nodes and concatenated.

    If ``full_info`` is True, the per-node distance-from-home (``batch.distances``) is appended as the last column.
    """
    x = torch.cat([batch.x_loc, batch.x_user_node], dim=-1)

    if use_rwpe:
        x = torch.cat([x, batch.random_walk_pe], dim=-1)

    if home_hop_distance is not None:
        home_pe = create_home_distance_encoding(
            batch, home_hop_distance, is_home_idx, n_bins=home_pe_bins, max_hops=home_pe_max_hops
        )
        x = torch.cat([x, home_pe], dim=-1)

    if use_demographics:
        x = torch.cat([x, batch.user_attr[batch.batch]], dim=-1)

    if not full_info:
        return x

    return torch.cat([x, batch.distances.to(x.dtype)], dim=-1)


def extracted_features_dim(
    dataset: VisitDataset,
    use_demographics: bool = True,
    full_info: bool = False,
    use_home_pe: bool = False,
    n_bins: int = HOME_PE_BINS,
    use_rwpe: bool = False,
) -> int:
    """Output width of extract_features for this dataset, given the same flags."""
    dim = dataset.num_features

    if use_rwpe:
        dim += dataset.num_rwpe_features
    if use_home_pe:
        dim += n_bins
    if use_demographics:
        dim += dataset.user_attributes.shape[1]
    if full_info:
        dim += dataset.distances.shape[-1]

    return dim


def _compute_node_index_within_graph(batch: UserBatch, device: torch.device) -> torch.Tensor:
    """Index of each node within its own graph (0..N-1), aligned to the node row order of the batch."""
    return torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]


def compute_home_hops(batch: UserBatch, home_hop_distance: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Hop distance from each node to the user's home node in the graph, shape ``[num_nodes]``."""
    device = batch.x_user_node.device
    node_idx = _compute_node_index_within_graph(batch, device)
    is_home = is_home_node_mask(batch.x_user_node, is_home_idx)

    home = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
    home[batch.batch[is_home]] = node_idx[is_home]

    return home_hop_distance[home[batch.batch], node_idx]


def create_home_distance_encoding(
    batch: UserBatch,
    home_hop_distance: torch.Tensor,
    is_home_idx: int,
    n_bins: int = HOME_PE_BINS,
    max_hops: int = HOME_PE_MAX_HOPS,
) -> torch.Tensor:
    """Positional encoding based on distance from user home (single anchor), shape ``[num_nodes, n_bins]``.

    RBF expansion of the number of hops to home for each node, capped at ``max_hops``, with ``n_bins`` centres spread
    evenly over ``[0, max_hops]``.
    """
    hops = compute_home_hops(batch, home_hop_distance, is_home_idx).clamp(max=max_hops)
    centers = torch.linspace(0, max_hops, n_bins, device=batch.x_user_node.device)

    return torch.exp(-((hops.unsqueeze(-1) - centers) ** 2) / 2.0)  # [num_nodes, n_bins]


@dataclass(frozen=True)
class RankingTensors:
    """Set of tensors read by the ranking metrics (R-Precision, Recall@K, etc).

    The tensors are restricted to the non-excluded/scored candidates (i.e. home node is excluded).
    """

    logits: Tensor
    probs: Tensor
    target: Tensor
    users: Tensor
    scored: Tensor


@dataclass(frozen=True)
class EvaluationTensors:
    """Set of tensors computed during the validation and testing steps.

    Metrics are computed using these tensors. All nodes are included in the tensors, whereas the ``ranking`` field
    contains the set of tensors with the home node excluded (used for the ranking metrics).
    """

    logits: Tensor
    loss: Tensor
    bce: Tensor
    probs: Tensor
    target: Tensor
    users: Tensor
    ranking: RankingTensors


class RankingModule(L.LightningModule):
    """LightningModule wrapping any GNN model for node-level binary prediction on activity graphs.

    Handles BCE or BPR training and per-epoch ranking metrics.
    Models must implement ``forward(x, edge_index, edge_attr, batch) -> logits``.

    Args:
        model: Any ``nn.Module`` with the GNN forward signature.
        lr: Initial learning rate for AdamW.
        loss: ``Loss`` instance.
        is_home_idx: Column index of ``is_home`` in ``batch.x_user_node``.
        full_info: If True, augment node features with home indicator and distances.
        use_demographics: concatenate the per-user demographics onto the node features.
        max_recall_k: Largest rank cutoff ``K`` of the headline ``avg_recall@K``.
        recall_ks: Rank cutoffs of the logged ``recall@k``.
        weight_decay: AdamW weight decay.
        lr_scheduler: Keyword arguments of a ReduceLROnPlateau scheduler on the loss's monitored metric. None trains
            without a scheduler.
        home_hop_distance: ``[num_nodes, num_nodes]`` contiguity-hop distance matrix, read for the per-positive hop
            distances and the home-anchored positional encodings.
        use_home_pe: use home-anchored positional encodings in features, default False.
        home_pe_bins: number of bins for the RBF expansion of the home PEs, default ``HOME_PE_BINS``.
        home_pe_max_hops: hop distance at which the home PEs are capped, default ``HOME_PE_MAX_HOPS``.
        calibration_bins: number of probability bins of the calibration error, default ``CALIBRATION_BINS``.
        compile_model: torch.compile the inner NN, default False.
        store_score_vectors: retain each test user's full per-node score vector.
        log_train_ranking: log ``train_avg_recall@K`` on the training batches, default True. Turn it off when per-user
            computation becomes expensive.
        use_rwpe: append the random-walk positional encoding to the node features, default False.
    """

    model: torch.nn.Module
    home_hop_distance: torch.Tensor  # registered buffer; annotated so it types as Tensor, not Tensor | Module

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        loss: Loss,
        is_home_idx: int,
        home_hop_distance: np.ndarray | torch.Tensor,
        max_recall_k: int,
        recall_ks: Sequence[int],
        full_info: bool = False,
        use_demographics: bool = True,
        weight_decay: float = 1e-4,
        lr_scheduler: dict | None = None,
        use_home_pe: bool = False,
        home_pe_bins: int = HOME_PE_BINS,
        home_pe_max_hops: int = HOME_PE_MAX_HOPS,
        calibration_bins: int = CALIBRATION_BINS,
        compile_model: bool = False,
        store_score_vectors: bool = False,
        log_train_ranking: bool = True,
        use_rwpe: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss", "home_hop_distance"])
        self.model = model
        self.lr = lr
        self.loss = loss
        self.full_info = full_info
        self.use_demographics = use_demographics
        self.max_recall_k = max_recall_k
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        # Metrics: General setup
        metrics = MetricCollection({
            "avg_recall": RetrievalAverageRecall(max_recall_k, recall_ks),
            "r_precision": RetrievalRPrecision(),
        })

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.log_train_ranking = log_train_ranking
        self.train_avg_recall = MeanMetric() if log_train_ranking else None

        self.val_calibration = BinaryCalibrationError(n_bins=calibration_bins, norm="l1")
        self.test_calibration = BinaryCalibrationError(n_bins=calibration_bins, norm="l1")

        self.is_home_idx = is_home_idx
        self.register_buffer(
            "home_hop_distance", torch.as_tensor(home_hop_distance, dtype=torch.float), persistent=False
        )

        # Metrics: per-user test scores retained for paired model comparison, in column format
        # (one entry per test user), populated by `on_test_epoch_end`.
        self.test_per_user = PerUserRanking(store_score_vectors=store_score_vectors)
        self.per_user_columns: dict[str, list] = {}

        # Full per-node score vector of each test user, kept only when `store_score_vectors` is set.
        self.per_user_score_vectors: list[list[float]] = []

        # Training: Home-anchored positional encodings injection
        self.use_home_pe = use_home_pe
        self.home_pe_bins = home_pe_bins
        self.home_pe_max_hops = home_pe_max_hops

        # Training: other positional encodings injection
        self.use_rwpe = use_rwpe

        # Training: torch.compile (applied in configure_model, not here)
        self.compile_model = compile_model
        self._compiled = False

    def configure_model(self) -> None:
        """Compile the inner NN with ``torch.compile``."""
        if self._compiled or not self.compile_model:
            return
        if self.trainer.fast_dev_run or self.trainer.overfit_batches:  # pyrefly: ignore[missing-attribute]
            return

        self.model.compile(dynamic=True)  # dynamic=True since graph sizes (i.e. batch sizes) differ
        self._compiled = True

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the wrapped model's logits for a batch's features."""
        return self.model(x, edge_index, edge_attr, batch)

    def on_after_batch_transfer(self, batch: pyg.data.Batch, dataloader_idx: int) -> UserBatch:
        """Return the typed view of each batch, which every step and callback then receives."""
        return UserBatch.from_pyg(batch)

    def training_step(self, batch: UserBatch, batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
        out = self.compute_logits(batch)
        loss = self.loss.loss_fn(out, batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        if self.train_avg_recall is not None:
            ranking_tensors = self._compute_ranking_tensors(batch, out)
            avg_recall, _, _ = per_user_recall(
                ranking_tensors.probs, ranking_tensors.target, ranking_tensors.users, self.max_recall_k, ()
            )
            self.train_avg_recall.update(avg_recall)
            self.log(
                f"train_avg_recall@{self.max_recall_k}",
                self.train_avg_recall,
                on_step=False,
                on_epoch=True,
                batch_size=batch.num_nodes,
            )

        return loss

    def _compute_ranking_tensors(self, batch: UserBatch, logits: Tensor) -> RankingTensors:
        scored_mask = ~is_home_node_mask(batch.x_user_node, self.is_home_idx)
        flat_logits = logits.squeeze(-1)

        return RankingTensors(
            logits=flat_logits[scored_mask],
            probs=flat_logits.sigmoid()[scored_mask],
            target=batch.y.squeeze(-1).long()[scored_mask],
            users=batch.user_id[batch.batch][scored_mask],
            scored=scored_mask,
        )

    def _common_evaluation_step(self, batch: UserBatch, step_prefix: Literal["val", "test"]) -> EvaluationTensors:
        out = self.compute_logits(batch)

        generator = torch.Generator(device=out.device).manual_seed(42)
        loss = self.loss.loss_fn(out, batch, generator)
        bce = F.binary_cross_entropy_with_logits(out, batch.y.float())
        probs = out.squeeze(-1).sigmoid()
        target = batch.y.squeeze(-1)
        users = batch.user_id[batch.batch]
        ranking_tensors = self._compute_ranking_tensors(batch, out)

        eval_tensors = EvaluationTensors(
            logits=out,
            loss=loss,
            bce=bce,
            probs=probs,
            target=target,
            users=users,
            ranking=ranking_tensors,
        )

        # Log loss metrics
        log_loss_kwargs = {"on_step": False, "on_epoch": True, "batch_size": batch.num_nodes}
        if self.loss.name != "bce":
            self.log(f"{step_prefix}_{self.loss.name}", eval_tensors.loss, **log_loss_kwargs)

        self.log(f"{step_prefix}_bce", eval_tensors.bce, **log_loss_kwargs)

        return eval_tensors

    def _num_scored_and_dropped_users(self, stage: Literal["val", "test"]) -> dict[str, float]:
        """Returns a dictionary of the number of scored and dropped users for a given stage.

        Each key is ``{stage}_{name}``, with name being either "n_scored_users" or "n_dropped_users".
        """
        collection = self.val_metrics if stage == "val" else self.test_metrics
        avg_recall = cast(RetrievalAverageRecall, collection[f"{stage}_avg_recall"])
        user_counts = avg_recall.user_counts()

        return {f"{stage}_{name}": value for name, value in user_counts.items()}

    def validation_step(self, batch: UserBatch, batch_idx: int) -> None:
        """Update the validation ranking, BCE and calibration metrics for one batch."""
        eval_tensors = self._common_evaluation_step(batch, "val")
        ranking_tensors = eval_tensors.ranking

        # Ranking metrics
        # Feed sigmoid (monotonic, preserves ranking), the same scores the per-user ranking and score vectors hold.
        self.val_metrics.update(ranking_tensors.probs, ranking_tensors.target, indexes=ranking_tensors.users)

        # Calibration metrics
        self.val_calibration.update(eval_tensors.probs, eval_tensors.target.long())
        self.log("val_calibration_l1", self.val_calibration, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Log and reset the validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("val"))
        self.val_metrics.reset()

    def test_step(self, batch: UserBatch, batch_idx: int) -> None:
        """Update the test metrics and collect per-user rankings and score vectors for one batch."""
        eval_tensors = self._common_evaluation_step(batch, "test")
        ranking = eval_tensors.ranking

        node_home_hops = compute_home_hops(batch, self.home_hop_distance, self.is_home_idx)

        # Ranking metrics
        self.test_metrics.update(  # Feed sigmoid (see validation step)
            ranking.probs, ranking.target, indexes=ranking.users
        )

        # Per-user metrics
        self.test_per_user.update(  # All nodes included, drops home node itself instead to allow full per-user comparison
            eval_tensors.probs,
            eval_tensors.target.long(),
            indexes=eval_tensors.users,
            exclude=~ranking.scored,
            hops=node_home_hops,
        )

        # Calibration metrics
        self.test_calibration.update(eval_tensors.probs, eval_tensors.target.long())
        self.log("test_calibration_l1", self.test_calibration, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log and reset the test metrics and keep the per-user columns and score vectors."""
        self.log_dict(self.test_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("test"))
        self.test_metrics.reset()

        # Not logged: these are per-user vectors, not scalars. Persisted by `run_experiment`.
        self.per_user_columns = {name: value.cpu().tolist() for name, value in self.test_per_user.columns().items()}
        self.per_user_columns = self.per_user_columns | self.test_per_user.positive_columns()
        self.per_user_score_vectors = self.test_per_user.score_vectors().cpu().tolist()
        self.test_per_user.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW, with a plateau LR scheduler when scheduling is on and not overfitting."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Only add a scheduler if not trying to overfit (i.e. not in diagnostic mode). Validation error is not a useful
        # signal when purposefully overfitting
        if self.trainer.overfit_batches == 0 and self.lr_scheduler is not None:  # pyrefly: ignore[missing-attribute]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.loss.monitor_mode, **self.lr_scheduler
            )

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": self.loss.monitor}}

        return optimizer

    def compute_logits(self, batch: UserBatch) -> torch.Tensor:
        """Model logits for a batch."""
        home_hop_distance = self.home_hop_distance if self.use_home_pe else None

        x = extract_features(
            batch,
            self.full_info,
            self.is_home_idx,
            use_demographics=self.use_demographics,
            home_hop_distance=home_hop_distance,
            home_pe_bins=self.home_pe_bins,
            home_pe_max_hops=self.home_pe_max_hops,
            use_rwpe=self.use_rwpe,
        )
        return self(x, batch.edge_index, batch.edge_attr, batch.batch)
