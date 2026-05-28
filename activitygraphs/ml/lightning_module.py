"""ActivityGraphModule and _EpochMetricsCallback for Lightning-based GNN training."""

import lightning as L
import torch
import torch.nn.functional as F
import torch_geometric as pyg

from activitygraphs.ml.experiment import extract_features
from activitygraphs.ml.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k



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
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.register_buffer("pos_weight", pos_weight)
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.full_info = full_info
        self.k = k
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bce"},
        }
