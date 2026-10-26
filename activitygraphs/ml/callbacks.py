"""Lightning callbacks for debugging overfit runs and collecting per-epoch metrics."""

import logging

import lightning as L
import torch

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.ml.lightning_module import RankingModule

log = logging.getLogger(__name__)


class OverfitDebugCallback(L.Callback):
    """Logs stats on the first batch and output stats on the last epoch's first batch."""

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: UserBatch, batch_idx: int
    ) -> None:
        """Log input statistics on the first epoch and output statistics on the last."""
        if batch_idx != 0:
            return

        assert isinstance(pl_module, RankingModule)

        if trainer.current_epoch == 0:
            b_min, b_max, b_mean = batch.x.min().item(), batch.x.max().item(), batch.x.mean().item()

            log.info("[overfit-debug] num_nodes=%d num_graphs=%d", batch.num_nodes, batch.num_graphs)
            log.info("[overfit-debug] y.sum()=%s y.numel()=%d", batch.y.sum().item(), batch.y.numel())
            log.info("[overfit-debug] x.isnan().any()=%s", torch.isnan(batch.x).any().item())
            log.info("[overfit-debug] x.min/max/mean=%.4f/%.4f/%.4f", b_min, b_max, b_mean)
            log.info("[overfit-debug] positive rate=%.4f", batch.y.float().mean().item())

        if trainer.max_epochs is not None and trainer.current_epoch == trainer.max_epochs - 1:
            with torch.no_grad():
                out = pl_module.compute_logits(batch)

            log.info(
                "[overfit-debug-final] out std=%.4f min=%.4f max=%.4f mean=%.4f",
                out.std().item(),
                out.min().item(),
                out.max().item(),
                out.mean().item(),
            )


class EpochMetricsCollector(L.Callback):
    """Collects per-epoch fit metrics and a final test row as stage-tagged dicts.

    Records one ``stage="fit"`` row per training epoch (``train_*``/``val_*`` metrics) and one
    ``stage="test"`` row per test run (``test_*`` metrics), avoiding the CSV write-and-reparse.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record the epoch's train and validation metrics as one row."""
        if trainer.sanity_checking:
            return

        metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith(("train_", "val_"))
        }
        if metrics:
            self.rows.append({"stage": "fit", "epoch": trainer.current_epoch, **metrics})

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record the test metrics as one row."""
        # Collect on `on_test_end`, not `on_test_epoch_end`: callback hooks run before the
        # LightningModule's `on_test_epoch_end`, which is where the Retrieval* ranking metrics are
        # logged. Reading at epoch-end would capture only the per-batch test_bce.
        metrics = {key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith("test_")}
        self.rows.append({"stage": "test", "epoch": None, **metrics})
