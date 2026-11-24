"""Experiment setup: the split, loss, baselines, architectures and training settings shared by an experiment's models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from activitygraphs.config import Config, ModelConfig
from activitygraphs.datasets import HouseholdSplit, SurveyDataModule
from activitygraphs.experiments.baselines import measure_baselines
from activitygraphs.experiments.restore import (
    RestoredVariant,
    SourceRun,
    build_restored_variants,
    check_checkpoints,
    check_node_weights,
    check_same_split,
    read_source_run,
    scaled_datamodules,
)
from activitygraphs.experiments.results import create_output_dir, run_config, save_config
from activitygraphs.experiments.training import WandBParams, home_distance_frame
from activitygraphs.experiments.variants import Variant, build_variants, check_distinct_names, with_overrides
from activitygraphs.runs import run_id
from activitygraphs.training.losses import Loss, build_loss


@dataclass(frozen=True)
class ExperimentSetup:
    """Shared experiment setup: dataset split, loss, baselines, and model hyperparameters.

    Built once per experiment by ``setup_experiment``, so all experiments train against the
    same split, loss, and baselines.

    Attributes:
        run_dir: Directory holding the config, results, checkpoints and logs written by the run. None on a
            ``fast_dev_run``, since a ``fast_dev_run`` writes nothing.
        variants: The variants trained from scratch, in training order.
        restored_variants: The variants restored from the recorded run of ``cfg.transfer.source_run``, in training
            order.
        scaled_datamodules: The datamodule of the experiment's users for each scaling of a restored variant, keyed by
            scaling. Empty with no recorded run.
    """

    cfg: Config
    run_dir: Path | None
    datamodule: SurveyDataModule
    loss: Loss
    models: dict[str, ModelConfig]
    variants: list[Variant]
    restored_variants: list[RestoredVariant]
    scaled_datamodules: dict[str, SurveyDataModule]
    baseline_results: list[pl.DataFrame]
    home_distances: pl.DataFrame
    train_seeds: list[int]
    debug: bool
    wandb_params: WandBParams


def setup_experiment(cfg: Config, with_baselines: bool = True) -> ExperimentSetup:
    """Set up an experiment.

    Configure logging, build the datamodule, loss, baselines, and the training settings shared by every model, and
    create the run directory with the config the run uses.

    The architectures and their hyperparameters are read from ``cfg.models``. When ``cfg.trainer.overfit_batches > 0``,
    every architecture's weight decay is set to 0 (and dropout too if it has some), and its learning rate to ``cfg.train.overfit_lr``,
    unless that is null, and ``cfg.trainer.max_epochs`` is set to ``cfg.train.overfit_epochs``. The split is drawn with ``cfg.train.split_seed`` and is never reseeded, so every model in
    every experiment is scored on the same test users.

    ``with_baselines=False`` skips the baselines. With ``cfg.transfer.fit_baselines_on_source`` on, the baselines
    are also fitted on the dataset of the recorded run of ``cfg.transfer.source_run``.

    The recorded run and the checkpoints of the restored variants are read and checked first, before any data is
    loaded.

    Raises:
        ValueError: One of the following holds.

            - Two variants share a name.
            - The recorded run is of this run's dataset and drew another split.
            - A variant with one weight per node is restored from a dataset with other nodes, or with its nodes in
              another order.
            - ``cfg.transfer.test_on_all_users`` is set and a model was trained or fitted on this run's dataset: a
              variant trained or fine-tuned here, a variant restored from a run of this dataset, or the baselines
              fitted on such a run.
    """
    split = HouseholdSplit(val_size=cfg.train.val_size, test_size=cfg.train.test_size)
    split_seed = cfg.train.split_seed
    train_seeds = list(cfg.train.train_seeds)
    runs_path = Path(cfg.paths.runs)

    overfitting = cfg.trainer.overfit_batches > 0
    models = dict(cfg.models)

    if overfitting:
        cfg = cast(Config, OmegaConf.merge(cfg, {"trainer": {"max_epochs": cfg.train.overfit_epochs}}))
        overrides: dict = {"weight_decay": 0.0}
        if cfg.train.overfit_lr is not None:
            overrides["lr"] = cfg.train.overfit_lr

        for key, model in models.items():
            has_dropout = "dropout" in cast(DictConfig, model)
            model_overrides = {**overrides, "dropout": 0.0} if has_dropout else overrides
            models[key] = with_overrides(model, model_overrides)

    transfer = cfg.transfer
    variants = build_variants(cfg.experiment.variants, models)
    source_run = read_source_run(runs_path, transfer)
    restored_variants = [] if source_run is None else build_restored_variants(transfer, source_run, cfg.data.name)

    check_checkpoints(restored_variants, train_seeds)
    if source_run is not None and source_run.dataset_name == cfg.data.name:
        check_same_split(source_run, split, split_seed)
    variant_names = [variant.name for variant in [*variants, *restored_variants]]
    check_distinct_names(variant_names)
    if transfer.test_on_all_users:
        _check_tested_across_datasets(cfg, variants, restored_variants, source_run)

    run = run_id(cfg.data.name, cfg.experiment.name, datetime.now())
    run_dir = create_output_dir(cfg, run)

    L.seed_everything(split_seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = SurveyDataModule(
        cfg,
        split,
        seed=split_seed,
        batch_size=cfg.train.loader.batch_size,
        num_workers=cfg.train.loader.num_workers,
        train_fraction=cfg.train.train_fraction,
        test_on_all_users=transfer.test_on_all_users,
    )
    datamodule.prepare_data()
    datamodule.setup()
    check_node_weights(restored_variants, datamodule.locations)

    datamodules = {} if source_run is None else scaled_datamodules(cfg, source_run, datamodule)
    loss = build_loss(cfg.train.loss, cfg.train.max_recall_k)

    if run_dir is not None:
        save_config(run_dir, cfg, models)

    use_wandb = cfg.train.wandb and not cfg.trainer.fast_dev_run
    wandb_params = WandBParams(
        use_wandb=use_wandb,
        project=cfg.train.wandb_project,
        entity=cfg.train.wandb_entity,
        group=run,
        dataset_name=cfg.data.name,
        experiment=cfg.experiment.name,
        config=run_config(cfg),
        run_dir=None if run_dir is None else str(run_dir),
    )

    baseline_results = []
    if with_baselines:
        visit_free_only = transfer.test_on_all_users
        baseline_results = measure_baselines(datamodule, loss, cfg, wandb_params, visit_free_only=visit_free_only)
    if with_baselines and source_run is not None and transfer.fit_baselines_on_source:
        loader = cfg.train.loader
        fit_datamodule = source_run.datamodule(loader.batch_size, loader.num_workers)
        name_suffix = f"-from{source_run.dataset_name}"
        baseline_results += measure_baselines(datamodule, loss, cfg, wandb_params, fit_datamodule, name_suffix)

    return ExperimentSetup(
        cfg=cfg,
        run_dir=run_dir,
        datamodule=datamodule,
        loss=loss,
        models=models,
        variants=variants,
        restored_variants=restored_variants,
        scaled_datamodules=datamodules,
        baseline_results=baseline_results,
        home_distances=home_distance_frame(datamodule),
        train_seeds=train_seeds,
        debug=overfitting or cfg.train.debug,
        wandb_params=wandb_params,
    )


def _check_tested_across_datasets(
    cfg: Config, variants: list[Variant], restored_variants: list[RestoredVariant], source_run: SourceRun | None
) -> None:
    """Check that every model tested on all users was trained or fitted on another dataset, and is only tested here.

    Raises:
        ValueError: A variant is trained or fine-tuned on this run's dataset, or restored from a run of it. Or the
            baselines are fitted on a recorded run of this run's dataset.
    """
    trained_names = [variant.name for variant in variants]
    for restored_variant in restored_variants:
        if restored_variant.fine_tune or not restored_variant.across_datasets:
            trained_names.append(restored_variant.name)

    if source_run is not None and cfg.transfer.fit_baselines_on_source and source_run.dataset_name == cfg.data.name:
        trained_names.append(f"the baselines fitted on {source_run.dataset_name}")

    if trained_names:
        raise ValueError(
            f"Testing on all users tests on the training users, so every model must be trained on another "
            f"dataset and only tested; {trained_names} are not."
        )
