"""Training orchestration: run_experiment, evaluate_baseline, and shared feature/weight helpers."""

from pathlib import Path

import lightning as L
import polars as pl
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.lightning_module import ActivityGraphModule, extract_features
from activitygraphs.ml.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k


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
    k: int = 5,
) -> pl.DataFrame:
    """Evaluate a baseline model and return a results DataFrame matching the ``run_experiment`` format."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    baseline = baseline.to(device)
    loss = _evaluate_bce(device, baseline, loader, full_info)
    loss_weight = _evaluate_bce(device, baseline, loader, full_info, pos_weight=pos_weight)
    metrics = _evaluate_at_k(device, baseline, loader, full_info, k=k)

    return pl.DataFrame({
        "name": [name],
        "epoch": [0],
        "train_loss": [0.0],
        "val_bce": [loss],
        "val_bce_weighted": [loss_weight],
        f"val_precision@{k}": [metrics[f"precision@{k}"]],
        f"val_recall@{k}": [metrics[f"recall@{k}"]],
        "val_mrr": [metrics["mrr"]],
        f"val_ndcg@{k}": [metrics[f"ndcg@{k}"]],
    })


def run_experiment(
    model: torch.nn.Module,
    datamodule: ActivityDataModule,
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    reg: str | None = None,
    full_info: bool = False,
    model_save_dir: Path | None = None,
    fast_dev_run: bool = False,
    overfit_batches: int = 0,
    weight_decay: float = 1e-4,
) -> pl.DataFrame:
    """Train a model and return per-epoch metrics as a Polars DataFrame.

    Uses AdamW with weight decay 1e-4 and ReduceLROnPlateau scheduling.

    Args:
        model: Model to train, must implement ``forward(x, edge_index, edge_attr, batch)``.
        datamodule: ``ActivityDataModule`` instance (``setup()`` is called internally if needed).
        num_epochs: Number of training epochs.
        verbose: Non-zero enables the Lightning progress bar.
        name: Experiment name used for logging and checkpoint filename.
        lr: Initial learning rate.
        reg: Optional regularisation; ``"l1"`` adds L1 weight penalty.
        full_info: Passed to ``extract_features`` to optionally include home/distance features.
        model_save_dir: Directory for checkpoint and CSV log files; None disables both.
        fast_dev_run: If True, runs 1 train batch and 1 val batch then exits; result DataFrame is empty.
        overfit_batches: Number of batches to overfit on; 0 disables (normal training).

    Returns:
        DataFrame with columns ``name``, ``epoch``, ``train_loss``, ``val_bce``,
        ``val_bce_weighted``, ``val_precision@5``, ``val_recall@5``, ``val_mrr``, ``val_ndcg@5``.
    """
    name = name or model.__class__.__name__

    datamodule.setup()

    lit_model = ActivityGraphModule(
        model=model,
        lr=lr,
        pos_weight=datamodule.pos_weight,
        reg=reg,
        full_info=full_info,
        weight_decay=weight_decay,
    )

    callbacks: list[L.Callback] = []

    if model_save_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(model_save_dir), filename=name, save_last=False, save_top_k=1, monitor="val_bce"
            )
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
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    if fast_dev_run:
        return pl.DataFrame()

    if model_save_dir is not None:
        trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    raw = pl.read_csv(logger.experiment.metrics_file_path).sort("step")
    metric_cols = [c for c in raw.columns if c not in ("epoch", "step")]
    return (
        raw
        .group_by("epoch")
        .agg([pl.col(c).drop_nulls().last().alias(c) for c in metric_cols])
        .sort("epoch")
        .with_columns(pl.lit(name).alias("name"))
        .select(["name", "epoch"] + metric_cols)
    )
