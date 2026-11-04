"""RankingModule and _EpochMetricsCallback for Lightning-based GNN training."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryCalibrationError

from activitygraphs.datasets.batch import UserBatch, UserRows
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.scorers import Scorer
from activitygraphs.training.losses import Loss
from activitygraphs.training.metrics import PerUserRanking, RetrievalAverageRecall, RetrievalRPrecision, per_user_recall

CALIBRATION_BINS = 15


@dataclass(frozen=True)
class RankingTensors:
    """Set of tensors read by the ranking metrics (R-Precision, Recall@K, etc).

    The tensors are restricted to the non-excluded/scored candidates (i.e. home node is excluded). Each tensor is flat,
    one entry per scored (user, node) pair, except ``scored``.

    Attributes:
        scored: Mask of the scored (user, node) pairs, flattened from ``[num_users, num_nodes]``.
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
    contains the set of tensors with the home node excluded (used for the ranking metrics). Every tensor but ``loss``
    and ``bce`` is flattened from ``[num_users, num_nodes]``, user by user.
    """

    logits: Tensor
    loss: Tensor
    bce: Tensor
    probs: Tensor
    target: Tensor
    users: Tensor
    ranking: RankingTensors


class RankingModule(L.LightningModule):
    """LightningModule wrapping any scorer for node-level binary prediction on activity graphs.

    Handles BCE or BPR training and per-epoch ranking metrics. Each batch of user rows is joined with the location
    graph on device, and every step receives the resulting ``UserBatch``. The logits read by the losses/the metrics are
    of shape ``[num_users, num_nodes]``.

    Args:
        scorer: The ``Scorer``, such as a ``NodeScorer``, a ``GraphScorer`` or a ``FittedScorer``, mapping the
            locations and the ``UserBatch`` to the logits.
        locations: The location graph joined with each batch of users, and passed to the scorer.
        lr: Initial learning rate for AdamW.
        loss: ``Loss`` instance.
        max_recall_k: Largest rank cutoff ``K`` of the headline ``avg_recall@K``.
        recall_ks: Rank cutoffs of the logged ``recall@k``.
        weight_decay: AdamW weight decay.
        lr_scheduler: Keyword arguments of a ReduceLROnPlateau scheduler on the loss's monitored metric. None trains
            without a scheduler.
        calibration_bins: number of probability bins of the calibration error, default ``CALIBRATION_BINS``.
        compile_model: torch.compile the trained network of the scorer, default False.
        store_score_vectors: retain each test user's full per-node score vector.
        log_train_ranking: log ``train_avg_recall@K`` on the training batches, default True. Turn it off when per-user
            computation becomes expensive.
    """

    scorer: Scorer

    def __init__(
        self,
        scorer: Scorer,
        locations: Locations,
        lr: float,
        loss: Loss,
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
        self.save_hyperparameters(ignore=["scorer", "locations", "loss"])
        self.scorer = scorer
        self.locations = locations
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
        """Compile the trained network of the scorer with ``torch.compile``."""
        if self._compiled or not self.compile_model:
            return
        if self.trainer.fast_dev_run or self.trainer.overfit_batches:  # pyrefly: ignore[missing-attribute]
            return

        self.scorer.compile_network()
        self._compiled = True

    def on_after_batch_transfer(self, batch: UserRows, dataloader_idx: int) -> UserBatch:
        """Join the on-device ``UserRows`` batch with the locations graph to return a ``UserBatch``.

        Lightning hook: triggers after batch has been loaded to device, subsequent steps and callbacks thus receive a
        ``UserBatch`` as input.
        """
        self.locations = self.locations.to(batch.home_node.device)

        return UserBatch.join(batch, self.locations)

    def training_step(self, batch: UserBatch, batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
        out = self.compute_logits(batch)
        loss = self.loss.loss_fn(out, batch)

        num_pairs = batch.num_users * batch.num_nodes
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=num_pairs)

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
                batch_size=num_pairs,
            )

        return loss

    def _compute_ranking_tensors(self, batch: UserBatch, logits: Tensor) -> RankingTensors:
        scored_mask = ~batch.is_home.reshape(-1)
        flat_logits = logits.reshape(-1)
        target = batch.visited.reshape(-1).long()
        users = _flat_user_idx(batch)

        return RankingTensors(
            logits=flat_logits[scored_mask],
            probs=flat_logits.sigmoid()[scored_mask],
            target=target[scored_mask],
            users=users[scored_mask],
            scored=scored_mask,
        )

    def _common_evaluation_step(self, batch: UserBatch, step_prefix: Literal["val", "test"]) -> EvaluationTensors:
        out = self.compute_logits(batch)

        generator = torch.Generator(device=out.device).manual_seed(42)
        loss = self.loss.loss_fn(out, batch, generator)
        bce = F.binary_cross_entropy_with_logits(out, batch.visited.float())
        flat_logits = out.reshape(-1)
        probs = flat_logits.sigmoid()
        target = batch.visited.reshape(-1)
        users = _flat_user_idx(batch)
        ranking_tensors = self._compute_ranking_tensors(batch, out)

        eval_tensors = EvaluationTensors(
            logits=flat_logits,
            loss=loss,
            bce=bce,
            probs=probs,
            target=target,
            users=users,
            ranking=ranking_tensors,
        )

        # Log loss metrics
        num_pairs = batch.num_users * batch.num_nodes
        log_loss_kwargs = {"on_step": False, "on_epoch": True, "batch_size": num_pairs}
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

        node_home_hops = batch.home_hops.reshape(-1)

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
        """Return the scorer's logits of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        return self.scorer(self.locations, batch)


def _flat_user_idx(batch: UserBatch) -> Tensor:
    """Return the dataset index of the user of each (user, node) pair, flattened from ``[num_users, num_nodes]``."""
    return batch.user_idx[:, None].expand(-1, batch.num_nodes).reshape(-1)
