"""ActivityGraphModule and _EpochMetricsCallback for Lightning-based GNN training."""

import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric as pyg

from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k


def extract_features(batch: pyg.data.Data | pyg.data.Batch, full_info: bool, use_demographics: bool = True):
    """Return node features from ``batch.x``. If `use_demographics` is true, then concatenates `batch.x` with
    `batch.graph_x`. If `full_info` is True, then it adds distances from home and the home features, but this is ONLY
     for the synthetic test case and can be disregarded."""
    x = batch.x

    if use_demographics and batch.graph_x is not None:
        if batch.batch is not None:
            demo = batch.graph_x[batch.batch]
        else:
            demo = batch.graph_x.expand(x.shape[0], -1)
        x = torch.cat([x, demo], dim=-1)

    if not full_info:
        return x

    raise ValueError("I only use this for the synthetic example in notebooks. Guard against accidental usage.")

    distances = batch.distances

    if batch.batch is not None:
        home_feature = batch.home_feature[batch.batch].unsqueeze(1)
    else:
        home_feature = torch.full((batch.x.shape[0], 1), batch.home_feature.item())

    return torch.cat([batch.x, home_feature, distances], dim=1)


def extracted_features_dim(dataset: ActivityDataset, use_demographics: bool = True) -> int:
    """Output width of extract_features for this dataset, given the same flags."""
    dim = dataset.num_features
    if use_demographics:
        dim += dataset.demographics.shape[1]
    return dim


class ActivityGraphModule(L.LightningModule):
    """LightningModule wrapping any GNN model for node-level binary prediction on activity graphs.

    Handles weighted BCE training, optional L1 regularisation, and per-epoch ranking metrics.
    Models must implement ``forward(x, edge_index, edge_attr, batch) -> logits``.

    Args:
        model: Any ``nn.Module`` with the GNN forward signature.
        lr: Initial learning rate for AdamW.
        pos_weight: Scalar positive-class weight for BCE loss, computed from the train split.
        reg: Optional regularisation type; only ``"l1"`` is supported.
        lambda_reg: L1 coefficient (ignored when ``reg`` is None).
        full_info: If True, augment node features with home indicator and distances.
        k: Rank cutoff for precision, recall, and NDCG metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        pos_weight: torch.Tensor,
        reg: str | None = None,
        lambda_reg: float = 0.01,
        full_info: bool = False,
        k: int = 5,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.register_buffer("pos_weight", pos_weight)
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.full_info = full_info
        self.k = k
        self.weight_decay = weight_decay
        self._val_outputs: list[dict] = []

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, batch: pyg.data.Batch, batch_idx: int) -> torch.Tensor:
        if self.current_epoch == 0 and batch_idx == 0:
            print(f"[overfit-debug] num_nodes={batch.num_nodes} num_graphs={batch.num_graphs}")
            print(f"[overfit-debug] y.sum()={batch.y.sum().item()} y.numel()={batch.y.numel()}")
            print(f"[overfit-debug] x.isnan().any()={torch.isnan(batch.x).any().item()}")
            print(
                f"[overfit-debug] x.min/max/mean={batch.x.min().item():.4f}/{batch.x.max().item():.4f}/{batch.x.mean().item():.4f}"
            )
            print(f"[overfit-debug] pos_weight={self.pos_weight.item()}")

        if self.current_epoch == self.trainer.max_epochs - 1 and batch_idx == 0:
            with torch.no_grad():
                out_debug = self(
                    extract_features(batch, self.full_info),
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                print(
                    f"[overfit-debug-final] out std={out_debug.std().item():.4f} "
                    f"min={out_debug.min().item():.4f} "
                    f"max={out_debug.max().item():.4f} "
                    f"mean={out_debug.mean().item():.4f}"
                )

        x = extract_features(batch, self.full_info)
        out = self(x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=self.pos_weight)

        if self.reg == "l1":
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + self.lambda_reg * l1_norm

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        return loss

    def validation_step(self, batch: pyg.data.Batch, batch_idx: int) -> None:
        x = extract_features(batch, self.full_info)
        out = self(x, batch.edge_index, batch.edge_attr, batch.batch)

        bce = F.binary_cross_entropy_with_logits(out, batch.y.float())
        bce_weighted = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=self.pos_weight)

        self.log("val_bce", bce, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        self.log("val_bce_weighted", bce_weighted, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        k = self.k
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            scores = out[mask].squeeze()
            labels = batch.y[mask].squeeze()

            if labels.sum().int().item() == 0:
                continue

            self._val_outputs.append({
                f"precision@{k}": precision_at_k(scores, labels, k),
                f"recall@{k}": recall_at_k(scores, labels, k),
                "mrr": mean_reciprocal_rank(scores, labels),
                f"ndcg@{k}": ndcg_at_k(scores, labels, k),
            })

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        k = self.k
        keys = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]

        for key in keys:
            mean_val = sum(d[key] for d in self._val_outputs) / len(self._val_outputs)
            self.log(f"val_{key}", mean_val)

        self._val_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bce"},
        }
