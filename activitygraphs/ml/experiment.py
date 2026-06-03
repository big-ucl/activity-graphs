"""Training orchestration: run_experiment, evaluate_baseline, and shared feature/weight helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from activitygraphs.ml.callbacks import EpochMetricsCollector, OverfitDebugCallback
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.lightning_module import ActivityGraphModule


@dataclass
class WandBParams:
    use_wandb: bool
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    dataset_name: str | None = None


def evaluate_baseline(
    baseline: torch.nn.Module,
    datamodule: ActivityDataModule,
    name: str,
    k: int = 5,
    wandb_params: WandBParams | None = None,
) -> pl.DataFrame:
    """Evaluate a baseline model and return a results DataFrame matching the ``run_experiment`` format.

    Returns two stage-tagged rows (one ``stage="fit"`` with the ``val_*`` metrics, one
    ``stage="test"`` with the ``test_*`` metrics) so baseline frames align with the
    ``run_experiment`` output.
    """
    baseline_module = ActivityGraphModule(
        model=baseline,
        lr=0.0,
        pos_weight=datamodule.pos_weight,
        k=k,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
    )

    if wandb_params and wandb_params.use_wandb:
        logger: WandbLogger | bool = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name,
            group=wandb_params.group,
            tags=[t for t in [wandb_params.dataset_name, "baseline"] if t],
        )
    else:
        logger = False

    trainer = L.Trainer(logger=logger, enable_progress_bar=False)

    try:
        (val_results,) = trainer.validate(baseline_module, datamodule=datamodule)
        (test_results,) = trainer.test(baseline_module, datamodule=datamodule)
    finally:
        if wandb_params and wandb_params.use_wandb:
            import wandb

            wandb.finish()

    fit_row = {"name": name, "stage": "fit", "epoch": 0, "train_loss": 0.0, **val_results}
    test_row = {"name": name, "stage": "test", "epoch": None, **test_results}
    return pl.DataFrame([fit_row, test_row])


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
    schedule_lr: bool = False,
    pop_mode: Literal["none", "offset", "feature"] = "none",
    wandb_params: WandBParams | None = None,
    debug: bool = False,
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
        weight_decay: Weight decay parameter to AdamW, defaults to 1e-4.
        schedule_lr: add a ReduceLROnPlateau scheduler to the optimizer, defaults to False.
        pop_mode: "none"=do not inject ``pop_logits``; "offset"=inject in the loss function, "feature"=inject as
            features to the model.
        wandb_params: parameters to configure WandB logging.
        debug: Flag that enables `OverfitDebugCallback` statistics printing at the start and end of training, defaults to False.

    Returns:
        Long-format DataFrame tagged by ``stage``: one ``stage="fit"`` row per epoch carrying
        ``train_loss`` and the ``val_*`` metrics, plus (when a test run executed) one
        ``stage="test"`` row carrying the ``test_*`` metrics. Columns: ``name``, ``stage``,
        ``epoch``, ``train_loss``, ``val_*``, ``test_*``; cells absent for a row's stage are null.
        Empty when ``fast_dev_run`` is True.
    """
    name = name or model.__class__.__name__

    datamodule.setup()

    pop_logit = datamodule.pop_logit if pop_mode != "none" else None

    lit_model = ActivityGraphModule(
        model=model,
        lr=lr,
        pos_weight=datamodule.pos_weight,
        reg=reg,
        full_info=full_info,
        k=datamodule.train_dataset.median_realised_size,
        weight_decay=weight_decay,
        schedule_lr=schedule_lr,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        pop_logit=pop_logit,
        pop_mode=pop_mode,
    )

    # Build the callbacks

    collector = EpochMetricsCollector()
    callbacks: list[L.Callback] = [collector, LearningRateMonitor(logging_interval="epoch")]

    if model_save_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(model_save_dir), filename=name, save_last=False, save_top_k=1, monitor="val_bce"
            )
        )

    if debug:
        callbacks.append(OverfitDebugCallback())

    log_dir = str(model_save_dir) if model_save_dir is not None else "."

    # Configure logging

    if wandb_params and wandb_params.use_wandb:
        logger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name,
            group=wandb_params.group,
            save_dir=log_dir,
            tags=[wandb_params.dataset_name] if wandb_params.dataset_name else None,
        )
        logger.log_hyperparams({
            "model": name,
            "dataset": wandb_params.dataset_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "reg": reg,
            "full_info": full_info,
            "schedule_lr": schedule_lr,
            "overfit_batches": overfit_batches,
            "pop_mode": pop_mode,
        })
    else:
        logger = CSVLogger(save_dir=log_dir, name=name)

    # Create trainer and fit

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

    # Process results

    try:
        if fast_dev_run:
            return pl.DataFrame()

        if model_save_dir is not None:
            trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

        results = pl.DataFrame(collector.rows).with_columns(pl.lit(name).alias("name"))

        first_cols = ["name", "stage", "epoch"]
        other_cols = [c for c in results.columns if c not in first_cols]
        results = results.select(first_cols + other_cols)

        return results
    finally:
        if wandb_params and wandb_params.use_wandb:
            import wandb

            wandb.finish()
