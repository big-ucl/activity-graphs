"""TwoTowerModule, the LightningModule of the models that score every node of a batch of users at once, and NodeBias."""

from collections.abc import Sequence
from typing import Literal, cast

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryCalibrationError

from activitygraphs.datasets.batch import UserBatch, UserRows
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.ml.lightning_module import CALIBRATION_BINS, EvaluationTensors, RankingTensors
from activitygraphs.ml.losses import Loss
from activitygraphs.ml.metrics import PerUserRanking, RetrievalAverageRecall, RetrievalRPrecision
from activitygraphs.ml.popularity import popularity_logit


class NodeBias(torch.nn.Module):
    """Scores every node of every user with a fixed per-node bias.

    At the home-excluded popularity logit of the training split, it ranks the nodes as ``VisitFrequencyBaseline`` does.

    Args:
        logits: Bias of each node, ``[num_nodes]``.
    """

    logits: Tensor

    def __init__(self, logits: Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    @classmethod
    def fit(cls, users: Users) -> "NodeBias":
        """Return the bias at the home-excluded popularity logit of ``users``."""
        return cls(popularity_logit(users.visited, users.home_node))

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the bias of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        return self.logits.expand(batch.num_users, -1)


class TwoTowerModule(L.LightningModule):
    """LightningModule for the models that score every node of a batch of users at once, from per-user rows.

    The model maps a ``UserBatch`` to logits ``[num_users, num_nodes]``. The metrics are those of
    ``RankingModule``, over the same candidates: every node but each user's own home node.

    Args:
        model: Module mapping a ``UserBatch`` to logits ``[num_users, num_nodes]``.
        locations: The location graph joined with each batch of users.
        lr: Initial learning rate for AdamW.
        loss: ``Loss`` over the grid of a ``UserBatch``.
        max_recall_k: Largest rank cutoff ``K`` of the headline ``avg_recall@K``.
        recall_ks: Rank cutoffs of the logged ``recall@k``.
        weight_decay: AdamW weight decay.
        calibration_bins: Number of probability bins of the calibration error.
        store_score_vectors: Retain each test user's full per-node score vector.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        locations: Locations,
        lr: float,
        loss: Loss[UserBatch],
        max_recall_k: int,
        recall_ks: Sequence[int],
        weight_decay: float,
        calibration_bins: int = CALIBRATION_BINS,
        store_score_vectors: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "locations", "loss"])
        self.model = model
        self.locations = locations
        self.lr = lr
        self.loss = loss
        self.max_recall_k = max_recall_k
        self.weight_decay = weight_decay

        metrics = MetricCollection({
            "avg_recall": RetrievalAverageRecall(max_recall_k, recall_ks),
            "r_precision": RetrievalRPrecision(),
        })
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.val_calibration = BinaryCalibrationError(n_bins=calibration_bins, norm="l1")
        self.test_calibration = BinaryCalibrationError(n_bins=calibration_bins, norm="l1")

        self.test_per_user = PerUserRanking(store_score_vectors=store_score_vectors)
        self.per_user_columns: dict[str, list] = {}
        self.per_user_score_vectors: list[list[float]] = []

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the model's logits for every node of every user of ``batch``, ``[num_users, num_nodes]``."""
        return self.model(batch)

    def on_after_batch_transfer(self, batch: UserRows, dataloader_idx: int) -> UserBatch:
        """Return the rows ``batch`` joined with the location graph, the ``UserBatch`` every step receives."""
        self.locations = self.locations.to(batch.home_node.device)

        return UserBatch.join(batch, self.locations)

    def training_step(self, batch: UserBatch, batch_idx: int) -> Tensor:
        """Compute and log the training loss for one batch."""
        logits = self(batch)
        loss = self.loss.loss_fn(logits, batch)

        num_rows = batch.num_users * batch.num_nodes
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=num_rows)

        return loss

    def _evaluation_tensors(self, batch: UserBatch, step_prefix: Literal["val", "test"]) -> EvaluationTensors:
        """Score ``batch``, log its losses, and return its tensors flattened to one row per (user, node) pair."""
        logits = self(batch)

        generator = torch.Generator(device=logits.device).manual_seed(42)
        loss = self.loss.loss_fn(logits, batch, generator)
        bce = F.binary_cross_entropy_with_logits(logits, batch.visited.float())

        scored = ~batch.is_home.reshape(-1)
        flat_logits = logits.reshape(-1)
        probs = flat_logits.sigmoid()
        target = batch.visited.reshape(-1)
        users = batch.user_idx[:, None].expand_as(logits).reshape(-1)

        ranking = RankingTensors(
            logits=flat_logits[scored],
            probs=probs[scored],
            target=target.long()[scored],
            users=users[scored],
            scored=scored,
        )

        num_rows = batch.num_users * batch.num_nodes
        log_loss_kwargs = {"on_step": False, "on_epoch": True, "batch_size": num_rows}
        if self.loss.name != "bce":
            self.log(f"{step_prefix}_{self.loss.name}", loss, **log_loss_kwargs)

        self.log(f"{step_prefix}_bce", bce, **log_loss_kwargs)

        return EvaluationTensors(
            logits=flat_logits, loss=loss, bce=bce, probs=probs, target=target, users=users, ranking=ranking
        )

    def _num_scored_and_dropped_users(self, stage: Literal["val", "test"]) -> dict[str, float]:
        """Return the number of scored and dropped users of ``stage``.

        The keys are ``{stage}_n_scored_users`` and ``{stage}_n_dropped_users``.
        """
        collection = self.val_metrics if stage == "val" else self.test_metrics
        avg_recall = cast(RetrievalAverageRecall, collection[f"{stage}_avg_recall"])
        user_counts = avg_recall.user_counts()

        return {f"{stage}_{name}": value for name, value in user_counts.items()}

    def validation_step(self, batch: UserBatch, batch_idx: int) -> None:
        """Update the validation ranking, BCE and calibration metrics for one batch."""
        eval_tensors = self._evaluation_tensors(batch, "val")
        ranking = eval_tensors.ranking

        self.val_metrics.update(ranking.probs, ranking.target, indexes=ranking.users)

        self.val_calibration.update(eval_tensors.probs, eval_tensors.target.long())
        self.log("val_calibration_l1", self.val_calibration, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Log and reset the validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("val"))
        self.val_metrics.reset()

    def test_step(self, batch: UserBatch, batch_idx: int) -> None:
        """Update the test metrics and collect per-user rankings and score vectors for one batch."""
        eval_tensors = self._evaluation_tensors(batch, "test")
        ranking = eval_tensors.ranking

        node_home_hops = batch.home_hops.reshape(-1)

        self.test_metrics.update(ranking.probs, ranking.target, indexes=ranking.users)

        self.test_per_user.update(
            eval_tensors.probs,
            eval_tensors.target.long(),
            indexes=eval_tensors.users,
            exclude=~ranking.scored,
            hops=node_home_hops,
        )

        self.test_calibration.update(eval_tensors.probs, eval_tensors.target.long())
        self.log("test_calibration_l1", self.test_calibration, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log and reset the test metrics and keep the per-user columns and score vectors."""
        self.log_dict(self.test_metrics.compute())
        self.log_dict(self._num_scored_and_dropped_users("test"))
        self.test_metrics.reset()

        self.per_user_columns = {name: value.cpu().tolist() for name, value in self.test_per_user.columns().items()}
        self.per_user_columns = self.per_user_columns | self.test_per_user.positive_columns()
        self.per_user_score_vectors = self.test_per_user.score_vectors().cpu().tolist()
        self.test_per_user.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW over the model's parameters."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
