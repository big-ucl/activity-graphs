import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import extract_features


class OverfitDebugCallback(L.Callback):
    """Prints stats on the first batch and output stats on the last epoch's first batch."""

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: pyg.data.Batch, batch_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        if trainer.current_epoch == 0:
            b_min, b_max, b_mean = batch.x.min().item(), batch.x.max().item(), batch.x.mean().item()

            print(f"[overfit-debug] num_nodes={batch.num_nodes} num_graphs={batch.num_graphs}")
            print(f"[overfit-debug] y.sum()={batch.y.sum().item()} y.numel()={batch.y.numel()}")
            print(f"[overfit-debug] x.isnan().any()={torch.isnan(batch.x).any().item()}")
            print(f"[overfit-debug] x.min/max/mean={b_min:.4f}/{b_max:.4f}/{b_mean:.4f}")
            print(f"[overfit-debug] pos_weight={pl_module.pos_weight.item()}")

        if trainer.max_epochs is not None and trainer.current_epoch == trainer.max_epochs - 1:
            with torch.no_grad():
                x = extract_features(batch, pl_module.full_info)
                out = pl_module(x, batch.edge_index, batch.edge_attr, batch.batch)

            print(
                f"[overfit-debug-final] out std={out.std().item():.4f} "
                f"min={out.min().item():.4f} max={out.max().item():.4f} mean={out.mean().item():.4f}"
            )
