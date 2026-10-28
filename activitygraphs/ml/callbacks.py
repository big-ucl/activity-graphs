"""Lightning callback for debugging overfit runs."""

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
            x = torch.cat([batch.x_loc, batch.x_user_node], dim=-1)
            b_min, b_max, b_mean = x.min().item(), x.max().item(), x.mean().item()

            log.info("[overfit-debug] num_nodes=%d num_graphs=%d", batch.num_nodes, batch.num_graphs)
            log.info("[overfit-debug] y.sum()=%s y.numel()=%d", batch.y.sum().item(), batch.y.numel())
            log.info("[overfit-debug] x.isnan().any()=%s", torch.isnan(x).any().item())
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
