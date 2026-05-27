"""Training orchestration: run_experiment, evaluate_baseline, and shared feature/weight helpers."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.logging import log

from activitygraphs.ml.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from activitygraphs.ml.datamodule import ActivityDataModule

def compute_training_weights(loader: pyg.loader.DataLoader) -> torch.Tensor:
    """Compute BCE positive-class weight as sqrt(neg_count / pos_count) over the full loader."""
    num_neg = torch.tensor(0, dtype=torch.float)
    num_pos = torch.tensor(0, dtype=torch.float)

    for batch in loader:
        num_neg += (batch.y == 0).sum()
        num_pos += batch.y.sum()

    weights = num_neg / num_pos
    return torch.sqrt(weights)


def extract_features(batch: pyg.data.Data | pyg.data.Batch, full_info: bool):
    """Return node features from ``batch.x``, optionally augmented with home features and distances."""
    if not full_info:
        return batch.x

    distances = batch.distances

    if batch.batch is not None:
        home_feature = batch.home_feature[batch.batch].unsqueeze(1)
    else:
        home_feature = torch.full((batch.x.shape[0], 1), batch.home_feature.item())

    return torch.cat([batch.x, home_feature, distances], dim=1)


@torch.no_grad()
def _evaluate_bce(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    full_info: bool,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """Return mean per-node BCE loss over the loader without gradients."""
    model.eval()
    epoch_loss = 0.0
    num_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)
        out = model(x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=pos_weight)
        epoch_loss += loss.item() * batch.num_nodes
        num_nodes += batch.num_nodes

    return epoch_loss / num_nodes


@torch.no_grad()
def _evaluate_at_k(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    full_info: bool,
    k: int = 5,
) -> dict:
    """Return mean precision@k, recall@k, MRR, and NDCG@k over all graphs in the loader."""
    model.eval()
    precisions, recalls, mrrs, ndcgs = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)
        out = model(x, batch.edge_index, batch.edge_attr, batch.batch)

        for i in range(batch.num_graphs):
            mask = batch.batch == i
            scores = out[mask].squeeze()
            labels = batch.y[mask].squeeze()

            if labels.sum().int().item() == 0:
                continue

            precisions.append(precision_at_k(scores, labels, k))
            recalls.append(recall_at_k(scores, labels, k))
            mrrs.append(mean_reciprocal_rank(scores, labels))
            ndcgs.append(ndcg_at_k(scores, labels, k))

    return {
        f"precision@{k}": sum(precisions) / len(precisions),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
    }


@torch.no_grad()
def evaluate_baseline(
    baseline: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    name: str,
    full_info: bool = False,
    pos_weight: torch.Tensor = None,
) -> dict:
    """Evaluate a baseline model and return a results dict matching the ``run_experiment`` format."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    baseline = baseline.to(device)
    loss = _evaluate_bce(device, baseline, loader, full_info)
    loss_weight = _evaluate_bce(device, baseline, loader, full_info, pos_weight=pos_weight)
    metrics = _evaluate_at_k(device, baseline, loader, full_info)

    k = 5
    log(
        Model=name,
        val_bce=loss,
        val_bce_weighted=loss_weight,
        precision_at_5=metrics[f"precision@{k}"],
        recall_at_5=metrics[f"recall@{k}"],
        mrr=metrics["mrr"],
        ndcg_at_5=metrics[f"ndcg@{k}"],
    )

    return {
        "name": name,
        "epoch": [0],
        "train_loss": [0.0],
        "val_bce": [loss],
        "val_bce_weighted": [loss_weight],
        f"val_precision@{k}": [metrics[f"precision@{k}"]],
        f"val_recall@{k}": [metrics[f"recall@{k}"]],
        "val_mrr": [metrics["mrr"]],
        f"val_ndcg@{k}": [metrics[f"ndcg@{k}"]],
    }


def run_experiment(
    model: torch.nn.Module,
    datamodule: ActivityDataModule
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    reg: str | None = None,
    full_info: bool = False,
    model_save_dir: Path | None = None,
) -> dict:
    """Train a model with Lightning and return per-epoch metrics as a dict.

    Uses AdamW with weight decay 1e-4 and ReduceLROnPlateau scheduling.

    Args:
        model: Model to train; must implement ``forward(x, edge_index, edge_attr, batch)``.
        datamodule: ``ActivityDataModule`` instance (``setup()`` is called internally if needed).
        num_epochs: Number of training epochs.
        verbose: Non-zero enables the Lightning progress bar.
        name: Experiment name used for logging and checkpoint filename.
        lr: Initial learning rate.
        reg: Optional regularisation; ``"l1"`` adds L1 weight penalty.
        full_info: Passed to ``extract_features`` to optionally include home/distance features.
        model_save_dir: Directory for checkpoint and CSV log files; None disables both.

    Returns:
        Dict with keys ``name``, ``epoch``, ``train_loss``, ``val_bce``,
        ``val_bce_weighted``, ``val_precision@5``, ``val_recall@5``, ``val_mrr``, ``val_ndcg@5``.
    """
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from activitygraphs.ml.lightning_module import ActivityGraphModule, _EpochMetricsCallback

    name = name or model.__class__.__name__

    datamodule.setup()

    lit_model = ActivityGraphModule(
        model=model,
        lr=lr,
        pos_weight=datamodule.pos_weight,
        reg=reg,
        full_info=full_info,
    )

    metrics_cb = _EpochMetricsCallback()
    callbacks: list[L.Callback] = [metrics_cb]

    if model_save_dir is not None:
        callbacks.append(
            ModelCheckpoint(dirpath=str(model_save_dir), filename=name, save_last=False, save_top_k=1, monitor="val_bce")
        )

    log_dir = str(model_save_dir) if model_save_dir is not None else "."
    logger = CSVLogger(save_dir=log_dir, name=name)

    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        gradient_clip_val=1.0,
        enable_progress_bar=bool(verbose),
        enable_model_summary=False,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    k = 5
    em = metrics_cb.epoch_metrics
    return {
        "name": name,
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": [m.get("train_loss", float("nan")) for m in em],
        "val_bce": [m.get("val_bce", float("nan")) for m in em],
        "val_bce_weighted": [m.get("val_bce_weighted", float("nan")) for m in em],
        f"val_precision@{k}": [m.get(f"val_precision@{k}", float("nan")) for m in em],
        f"val_recall@{k}": [m.get(f"val_recall@{k}", float("nan")) for m in em],
        "val_mrr": [m.get("val_mrr", float("nan")) for m in em],
        f"val_ndcg@{k}": [m.get(f"val_ndcg@{k}", float("nan")) for m in em],
    }
