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
from activitygraphs.datasets.dataset import is_home_node_mask
from activitygraphs.ml.baselines.common import Baseline
from activitygraphs.ml.features import FeatureEncoder, compute_home_hops
from activitygraphs.ml.losses import Loss
from activitygraphs.ml.metrics import PerUserRanking, RetrievalAverageRecall, RetrievalRPrecision, per_user_recall

CALIBRATION_BINS = 15


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
    Models must implement ``forward(x, edge_index, edge_attr, batch) -> logits``, except a ``Baseline``, which is
    given the batch itself.

    Args:
        model: Any ``nn.Module`` with the GNN forward signature, or a ``Baseline``.
        feature_encoder: Builds the input ``x`` to the ``model`` from the batch. None for a ``Baseline``.
        lr: Initial learning rate for AdamW.
        loss: ``Loss`` instance.
        is_home_idx: Column index of ``is_home`` in ``batch.x_user_node``.
        max_recall_k: Largest rank cutoff ``K`` of the headline ``avg_recall@K``.
        recall_ks: Rank cutoffs of the logged ``recall@k``.
        weight_decay: AdamW weight decay.
        lr_scheduler: Keyword arguments of a ReduceLROnPlateau scheduler on the loss's monitored metric. None trains
            without a scheduler.
        home_hop_distance: ``[num_nodes, num_nodes]`` contiguity-hop distance matrix, read for the per-positive hop
            distances.
        calibration_bins: number of probability bins of the calibration error, default ``CALIBRATION_BINS``.
        compile_model: torch.compile the inner NN, default False.
        store_score_vectors: retain each test user's full per-node score vector.
        log_train_ranking: log ``train_avg_recall@K`` on the training batches, default True. Turn it off when per-user
            computation becomes expensive.

    Raises:
        ValueError: ``encoder`` is None for a model that is not a ``Baseline``, or given for a ``Baseline``.
    """

    model: torch.nn.Module
    home_hop_distance: torch.Tensor

    def __init__(
        self,
        model: torch.nn.Module,
        feature_encoder: FeatureEncoder | None,
        lr: float,
        loss: Loss,
        is_home_idx: int,
        home_hop_distance: np.ndarray | torch.Tensor,
        max_recall_k: int,
        recall_ks: Sequence[int],
        weight_decay: float = 1e-4,
        lr_scheduler: dict | None = None,
        calibration_bins: int = CALIBRATION_BINS,
        compile_model: bool = False,
        store_score_vectors: bool = False,
        log_train_ranking: bool = True,
    ):
        super().__init__()
        if (feature_encoder is None) != isinstance(model, Baseline):
            raise ValueError("Every model but a `Baseline` takes a `FeatureEncoder`, and a `Baseline` takes none.")

        self.save_hyperparameters(ignore=["model", "encoder", "loss", "home_hop_distance"])
        self.model = model
        self.encoder = feature_encoder
        self.lr = lr
        self.loss = loss
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

        # Metrics: Full per-node score vector of each test user, kept only when `store_score_vectors` is set.
        self.per_user_score_vectors: list[list[float]] = []

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
        if self.encoder is None:
            return self.model(batch)

        x = self.encoder(batch)
        return self(x, batch.edge_index, batch.edge_attr, batch.batch)
