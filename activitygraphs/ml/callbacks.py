import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule, extract_features


class OverfitDebugCallback(L.Callback):
    """Prints stats on the first batch and output stats on the last epoch's first batch."""

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: pyg.data.Batch, batch_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        assert isinstance(pl_module, ActivityGraphModule)

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
                out = pl_module.compute_logits(x, batch.edge_index, batch.edge_attr, batch.batch)

            print(
                f"[overfit-debug-final] out std={out.std().item():.4f} "
                f"min={out.min().item():.4f} max={out.max().item():.4f} mean={out.mean().item():.4f}"
            )


class EpochMetricsCollector(L.Callback):
    """Collects per-epoch fit metrics and a final test row as stage-tagged dicts.

    Records one ``stage="fit"`` row per training epoch (``train_*``/``val_*`` metrics) and one
    ``stage="test"`` row per test run (``test_*`` metrics), avoiding the CSV write-and-reparse.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith(("train_", "val_"))
        }
        if metrics:
            self.rows.append({"stage": "fit", "epoch": trainer.current_epoch, **metrics})

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Collect on `on_test_end`, not `on_test_epoch_end`: callback hooks run before the
        # LightningModule's `on_test_epoch_end`, which is where the Retrieval* ranking metrics are
        # logged. Reading at epoch-end would capture only the per-batch test_bce/test_bce_weighted.
        metrics = {key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith("test_")}
        self.rows.append({"stage": "test", "epoch": None, **metrics})
