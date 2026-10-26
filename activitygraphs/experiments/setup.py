"""Experiment setup: the split, loss, baselines, architectures and training settings shared by an experiment's models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import torch
from omegaconf import OmegaConf

from activitygraphs.config import Config, ModelConfig
from activitygraphs.datasets import HouseholdSplit, VisitDataModule
from activitygraphs.experiments.baselines import measure_baselines
from activitygraphs.experiments.results import create_output_dir, run_config, save_config
from activitygraphs.experiments.training import WandBParams, home_distance_frame
from activitygraphs.ml.losses import Loss, build_loss
from activitygraphs.runs import TIMESTAMP_FORMAT


@dataclass(frozen=True)
class ExperimentSetup:
    """Shared experiment setup: dataset split, loss, baselines, and model hyperparameters.

    Built once per experiment by ``setup_experiment``, so all experiments train against the
    same split, loss, and baselines.

    Attributes:
        run_dir: Directory holding the config, results, checkpoints and logs written by the run. None on a
            ``fast_dev_run``, since a ``fast_dev_run`` writes nothing.
    """

    cfg: Config
    run_dir: Path | None
    datamodule: VisitDataModule
    loss: Loss
    models: dict[str, ModelConfig]
    baseline_results: list[pl.DataFrame]
    home_distances: pl.DataFrame
    train_seeds: list[int]
    debug: bool
    wandb_params: WandBParams


def with_overrides(model: ModelConfig, overrides: dict) -> ModelConfig:
    """Return a copy of ``model`` with ``overrides`` merged in; an interpolated name follows the new values."""
    return cast(ModelConfig, OmegaConf.merge(model, overrides))


def setup_experiment(cfg: Config, with_baselines: bool = True) -> ExperimentSetup:
    """Set up an experiment.

    Configure logging, build the datamodule, loss, baselines, and the training settings shared by every model, and
    create the run directory with the config the run uses.

    The architectures and their hyperparameters are read from ``cfg.models``. When ``cfg.trainer.overfit_batches > 0``,
    every architecture's dropout and weight decay are set to 0 and its learning rate to ``cfg.train.overfit_lr``,
    unless that is null, and ``cfg.trainer.max_epochs`` is set to ``cfg.train.overfit_epochs``. The split is drawn with ``cfg.train.split_seed`` and is never reseeded, so every model in
    every experiment is scored on the same test users.

    ``with_baselines=False`` skips fitting the frequency baselines.
    """
    started = datetime.now()
    run_dir = create_output_dir(cfg, started)

    split = HouseholdSplit(val_size=cfg.train.val_size, test_size=cfg.train.test_size)
    split_seed = cfg.train.split_seed
    train_seeds = list(cfg.train.train_seeds)

    L.seed_everything(split_seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = VisitDataModule(
        cfg,
        split,
        seed=split_seed,
        batch_size=cfg.train.loader.batch_size,
        num_workers=cfg.train.loader.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup()

    train_dataset = datamodule.train_dataset

    home_hop_distance = torch.as_tensor(train_dataset.home_hop_distance, dtype=torch.float)
    loss = build_loss(cfg.train.loss, home_hop_distance, train_dataset.is_home_col_idx, cfg.train.max_recall_k)

    overfitting = cfg.trainer.overfit_batches > 0
    models = dict(cfg.models)

    if overfitting:
        cfg = cast(Config, OmegaConf.merge(cfg, {"trainer": {"max_epochs": cfg.train.overfit_epochs}}))
        overrides: dict = {"weight_decay": 0.0, "dropout": 0.0}
        if cfg.train.overfit_lr is not None:
            overrides["lr"] = cfg.train.overfit_lr
        models = {key: with_overrides(model, overrides) for key, model in models.items()}

    if run_dir is not None:
        save_config(run_dir, cfg, models)

    wandb_params = WandBParams(
        use_wandb=cfg.train.wandb,
        project=cfg.train.wandb_project,
        entity=cfg.train.wandb_entity,
        group=f"{cfg.data.name}-{cfg.train.experiment}-{started:{TIMESTAMP_FORMAT}}",
        dataset_name=cfg.data.name,
        experiment=cfg.train.experiment,
        config=run_config(cfg),
        run_dir=None if run_dir is None else str(run_dir),
    )

    num_nodes = train_dataset.num_nodes
    baseline_results = measure_baselines(num_nodes, datamodule, loss, cfg, wandb_params) if with_baselines else []

    return ExperimentSetup(
        cfg=cfg,
        run_dir=run_dir,
        datamodule=datamodule,
        loss=loss,
        models=models,
        baseline_results=baseline_results,
        home_distances=home_distance_frame(datamodule),
        train_seeds=train_seeds,
        debug=overfitting or cfg.train.debug,
        wandb_params=wandb_params,
    )
