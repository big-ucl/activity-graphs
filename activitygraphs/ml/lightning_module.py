"""ActivityGraphModule and _EpochMetricsCallback for Lightning-based GNN training."""

from collections.abc import Collection
from typing import Literal, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection, SumMetric
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalPrecision, RetrievalRecall

from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.metrics import DEFAULT_HOP_BANDS, build_hop_band_metrics, RetrievalRPrecision
from activitygraphs.ml.sampling import poisson_sampling


def extract_features(
    batch: pyg.data.Data | pyg.data.Batch,
    full_info: bool,
    use_demographics: bool = True,
    pop_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return node features for a batch.

    Starts with ``batch.x`` (which already contains the network features and the per-user
    ``is_home`` indicator). If ``use_demographics`` is True, the per-graph user-specific demographics
    (``batch.graph_x``) are broadcast over nodes and concatenated. If ``full_info`` is True,
    the per-node distance-from-home (``batch.distances``) is appended as the last column;
    this is used only by the distance-augmented MLP baseline so that the explicit
    distance-to-home signal does not leak into the other models.
    """
    x = cast(torch.Tensor, batch.x)

    if pop_logit is not None:
        pop_logit_col = create_pop_logit_column(pop_logit, batch, standardize=True)
        x = torch.cat([x, pop_logit_col], dim=-1)

    if use_demographics and batch.graph_x is not None:
        if batch.batch is not None:
            demo = batch.graph_x[batch.batch]
        else:
            demo = batch.graph_x.expand(x.shape[0], -1)
        x = torch.cat([x, demo], dim=-1)

    if not full_info:
        return x

    return torch.cat([x, batch.distances.to(x.dtype)], dim=-1)


def extracted_features_dim(
    dataset: ActivityDataset, use_demographics: bool = True, full_info: bool = False, use_pop_feature: bool = False
) -> int:
    """Output width of extract_features for this dataset, given the same flags."""
    dim = dataset.num_features

    if use_pop_feature:
        dim += 1
    if use_demographics:
        dim += dataset.demographics.shape[1]
    if full_info:
        dim += dataset.distances.shape[-1]

    return dim


def create_pop_logit_column(pop_logit: torch.Tensor, batch, standardize: bool):
    """Per-node popularity logit aligned to the nodes in `batch`, shape [num_nodes, 1].

    `standardize` z-scores it over the node vector (for the pop-logit-as-input-feature variant).
    """
    pop_logit_col = pop_logit[_compute_node_index_within_graph(batch, pop_logit.device)]
    if standardize:
        pop_logit_col = (pop_logit_col - pop_logit.mean()) / pop_logit.std().clamp(min=1e-6)

    return pop_logit_col.unsqueeze(-1)


def _compute_node_index_within_graph(batch, device: torch.device) -> torch.Tensor:
    """Index of each node within its own graph (0..N-1), aligned to batch.x row order."""
    n = batch.num_nodes
    if getattr(batch, "batch", None) is None:
        return torch.arange(n, device=device)

    return torch.arange(n, device=device) - batch.ptr.to(device)[batch.batch]


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
        weight_decay: AdamW weight decay.
        schedule_lr: add a ReduceLROnPlateau scheduler to the optimizer, defaults to False.
        home_hop_distance: Optional ``[num_nodes, num_nodes]`` contiguity-hop distance matrix
            enabling distance-from-home hop-band ranking metrics at test time. Requires ``is_home_idx``.
        is_home_idx: Column index of ``is_home`` in the node feature matrix (for the hop-band metrics).
        hop_bands: Hop-distance bands for the hop-band metrics.
        pop_logit: Logits of global per-node visit frequencies.
        pop_mode: "none"=do not inject ``pop_logits``; "offset"=inject in the loss function, "feature"=inject as
            features to the model.
    """

    pos_weight: torch.Tensor  # registered buffer; annotated so it types as Tensor, not Tensor | Module

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
        schedule_lr: bool = False,
        home_hop_distance: np.ndarray | torch.Tensor | None = None,
        is_home_idx: int | None = None,
        hop_bands: Collection[tuple[str, float, float]] = DEFAULT_HOP_BANDS,
        pop_logit: torch.Tensor | None = None,
        pop_mode: Literal["none", "offset", "feature"] = "none",
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
        self.schedule_lr = schedule_lr

        # Metrics: General setup
        metrics = MetricCollection({
            "r_precision": RetrievalRPrecision(),
            f"recall@{k}": RetrievalRecall(top_k=k, empty_target_action="skip"),
            f"ndcg@{k}": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            f"precision@{k}": RetrievalPrecision(top_k=k, empty_target_action="skip"),
        })

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.val_calibration = BinaryCalibrationError(n_bins=15, norm="l1")
        self.test_calibration = BinaryCalibrationError(n_bins=15, norm="l1")

        # Metrics: Optional distance-from-home hop-band metrics (test time only).
        self.is_home_idx = is_home_idx
        self.hop_bands = hop_bands
        if home_hop_distance is not None:
            if is_home_idx is None:
                raise ValueError("is_home_idx is required when home_hop_distance is provided.")
            self.register_buffer(
                "home_hop_distance", torch.as_tensor(home_hop_distance, dtype=torch.float), persistent=False
            )
            self.hop_band_test_metrics = build_hop_band_metrics(hop_bands, k)
            # Per-band weighted BCE (node-pooled) and visited-node count, for context alongside the ranking bands.
            self.hop_band_bce = torch.nn.ModuleDict({label: MeanMetric() for label, _, _ in hop_bands})
            self.hop_band_pos = torch.nn.ModuleDict({label: SumMetric() for label, _, _ in hop_bands})
            self._seen_hop_bands: set[str] = set()
        else:
            self.home_hop_distance = None
            self.hop_band_test_metrics = None
            self.hop_band_bce = None
            self.hop_band_pos = None

        # Metrics: Capture predicted and true expected |RG_i| sizes
        self.test_pred_size = MeanMetric()
        self.test_true_size = MeanMetric()

        # Metrics: evaluate actual Poisson-sampled sets
        self.test_sampled_recall = MeanMetric()
        self.test_sampled_size = MeanMetric()

        # Training: Population logits injection
        if pop_mode not in ("none", "offset", "feature"):
            raise ValueError(f"pop_mode must be none|offset|feature, got {pop_mode}")
        if pop_mode != "none" and pop_logit is None:
            raise ValueError(f"pop_mode must be none|offset, got {pop_mode}")

        self.pop_mode = pop_mode
        if pop_logit is not None:
            self.register_buffer("pop_logit", pop_logit, persistent=False)
        else:
            self.pop_logit = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        out = self.compute_logits(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())

        if self.reg == "l1":
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + self.lambda_reg * l1_norm

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        return loss

    def _common_val_test_step(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        out = self.compute_logits(batch)
        bce = F.binary_cross_entropy_with_logits(out, batch.y.float())
        bce_weighted = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=self.pos_weight)
        probs = out.squeeze(-1).sigmoid()
        target = batch.y.squeeze(-1)
        users = batch.user_id[batch.batch]

        return out, bce, bce_weighted, probs, target, users

    def validation_step(self, batch, batch_idx: int) -> None:
        out, bce, bce_weighted, probs, target, users = self._common_val_test_step(batch)

        self.log("val_bce", bce, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        self.log("val_bce_weighted", bce_weighted, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        # torchmetrics Retrieval* treat preds as probabilities and drop preds <= 0, so feed sigmoid
        # (monotonic, preserves ranking) rather than raw logits.
        self.val_metrics.update(out.squeeze(-1).sigmoid(), target.long(), indexes=users)

        self.val_calibration.update(probs, target.long())

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        self.log("val_calibration_l1", self.val_calibration.compute())
        self.val_calibration.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        out, bce, bce_weighted, probs, target, users = self._common_val_test_step(batch)

        # Log BCEs
        self.log("test_bce", bce, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        self.log("test_bce_weighted", bce_weighted, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        # Update other metrics, feed sigmoid to conform to torchmetrics calling convention (see validation step)
        self.test_metrics.update(out.squeeze(-1).sigmoid(), batch.y.squeeze(-1).long(), indexes=users)

        # Update hop-band metrics
        if self.hop_band_test_metrics is not None:
            self._update_hop_band_metrics(batch, out)

        # Update BCE calibration
        self.test_calibration.update(probs, target.long())

        # Update true and predicted expected |RG_i| set sizes
        for idx in torch.unique(batch.user_id):
            m = users == idx
            self.test_pred_size.update(probs[m].sum())
            self.test_true_size.update(target[m].sum())

        # Update sampled set recall and size metrics
        self._update_sampled_sets_metrics(batch, out)

    def _update_hop_band_metrics(self, batch: pyg.data.Batch, out: torch.Tensor) -> None:
        """Update the hop-band ranking metrics, grouping nodes by hop distance from each user's home."""
        device = out.device
        logits = out.squeeze(-1)
        scores = logits.sigmoid()
        target = batch.y.squeeze(-1).long()

        # Each node's position within its graph is its network node index (graphs share the network order).
        node_idx = torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]
        is_home = batch.x[:, self.is_home_idx] > 0.0
        home_idx_per_graph = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
        home_idx_per_graph[batch.batch[is_home]] = node_idx[is_home]

        hop = self.home_hop_distance[home_idx_per_graph[batch.batch], node_idx]
        user_ids = batch.user_id[batch.batch]

        for label, low, high in self.hop_bands:
            mask = (hop >= low) & (hop <= high)
            if not mask.any():
                continue
            band_target = target[mask]
            self.hop_band_test_metrics[label].update(scores[mask], band_target, indexes=user_ids[mask])

            band_bce = F.binary_cross_entropy_with_logits(logits[mask], band_target.float(), reduction="none")
            self.hop_band_bce[label].update(band_bce)
            self.hop_band_pos[label].update(band_target.sum())

            if band_target.sum() > 0:
                self._seen_hop_bands.add(label)

    def _update_sampled_sets_metrics(self, batch, out, n_draws: int = 8):
        logits = out.squeeze(-1)
        target = batch.y.squeeze(-1).float()
        users = batch.user_id[batch.batch]
        gen = torch.Generator(device=out.device).manual_seed(42)

        # pos_weight=None under plain-BCE training (the default); pass float(self.pos_weight) only
        # if the model was trained with weighted BCE.
        for _ in range(n_draws):
            chosen = poisson_sampling(logits, gen, pos_weight=None).squeeze(-1)
            for idx in torch.unique(users):
                m = users == idx
                r = target[m].sum()
                if r == 0:
                    continue
                hits = (chosen[m] * target[m]).sum()
                self.test_sampled_recall.update(hits / r)
                self.test_sampled_size.update(chosen[m].sum())

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

        if self.hop_band_test_metrics is not None:
            # Only compute/log hop bands that saw at least one positive (others would have all groups skipped).
            for label in self._seen_hop_bands:
                self.log_dict(self.hop_band_test_metrics[label].compute())
                self.log(f"test_hop_{label}_nll", self.hop_band_bce[label].compute())
                self.log(f"test_hop_{label}_n_pos", self.hop_band_pos[label].compute())
            for collection in self.hop_band_test_metrics.values():
                collection.reset()
            for metric in self.hop_band_bce.values():
                metric.reset()
            for metric in self.hop_band_pos.values():
                metric.reset()
            self._seen_hop_bands.clear()

        self.log("test_calibration_l1", self.test_calibration.compute())
        self.test_calibration.reset()

        self.log("test_pred_size", self.test_pred_size.compute())
        self.log("test_true_size", self.test_true_size.compute())
        self.test_pred_size.reset()
        self.test_true_size.reset()

        self.log("test_sampled_recall", self.test_sampled_recall.compute())
        self.log("test_sampled_size", self.test_sampled_size.compute())
        self.test_sampled_recall.reset()
        self.test_sampled_size.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Only add a scheduler if not trying to overfit (i.e. not in diagnostic mode). Validation error is not a useful
        # signal when purposefully overfitting
        if self.trainer.overfit_batches == 0 and self.schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bce"}}

        return optimizer

    def _pop_logits_feature(self) -> torch.Tensor | None:
        return self.pop_logit if self.pop_mode == "feature" else None

    def compute_logits(self, batch) -> torch.Tensor:
        """Model logits for a batch, includes node-popularity feature column or offset if configured."""
        x = extract_features(batch, self.full_info, pop_logit=self._pop_logits_feature())
        out = self(x, batch.edge_index, batch.edge_attr, batch.batch)

        if self.pop_mode == "offset":
            out = out + create_pop_logit_column(self.pop_logit, batch, standardize=False)

        return out
