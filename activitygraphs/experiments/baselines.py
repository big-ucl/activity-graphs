"""Baseline runner: fit every baseline on the training split and evaluate it on the validation and test splits."""

from dataclasses import dataclass

import polars as pl
import torch

from activitygraphs.config import Config
from activitygraphs.datasets import VisitDataModule
from activitygraphs.experiments.training import WandBParams, evaluate_baseline
from activitygraphs.ml.baselines import (
    ConditionalGravityBaseline,
    ConditionalVisitFrequencyBaseline,
    DistanceDecayBaseline,
    GlobalBaseline,
    GravityBaseline,
    GravityBinnedBaseline,
    HomeZoneMFBaseline,
    UniformBaseline,
    VisitFrequencyBaseline,
)
from activitygraphs.ml.losses import Loss


@dataclass(frozen=True)
class BaselineSpec:
    """A fitted baseline, the name it is reported under, and how it is evaluated."""

    name: str
    baseline: torch.nn.Module
    full_info: bool = False


def measure_baselines(
    num_nodes,
    datamodule: VisitDataModule,
    loss: Loss,
    cfg: Config,
    wandb_params: WandBParams,
):
    """Fit and evaluate the baselines: the frequency baselines, then the distance-decay ones, then HomeZoneMF.

    The closed-form baselines are fitted once and their frames carry a null ``seed``. ``HomeZoneMF`` is fitted by
    stochastic optimisation, so it is refitted once per training seed in ``cfg.train.train_seeds`` and its frames
    carry that seed.

    Returns:
        One result frame per baseline, in order, then one per training seed for ``HomeZoneMF``.
    """
    datamodule.setup()
    max_recall_k = cfg.train.max_recall_k
    train_seeds = list(cfg.train.train_seeds)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    is_home_idx = datamodule.train_dataset.is_home_col_idx
    distance_scaler = datamodule.scalers.distances

    gravity = GravityBaseline(is_home_idx, distance_scaler).fit(train_loader)
    gravity_binned = GravityBinnedBaseline(is_home_idx, distance_scaler).fit(train_loader)
    conditional_visit_frequency = ConditionalVisitFrequencyBaseline(num_nodes, is_home_idx)
    conditional_gravity = ConditionalGravityBaseline(gravity)

    baselines = [
        BaselineSpec("Uniform", UniformBaseline()),
        BaselineSpec("GlobalMarginal", GlobalBaseline().fit(train_loader)),
        BaselineSpec("VisitFrequency", VisitFrequencyBaseline(num_nodes, is_home_idx).fit(train_loader)),
        BaselineSpec("ConditionalVisitFrequency", conditional_visit_frequency.fit(train_loader, val_loader)),
        BaselineSpec("DistanceDecay", DistanceDecayBaseline(distance_scaler), full_info=True),
        BaselineSpec("Gravity", gravity, full_info=True),
        BaselineSpec("GravityBinned", gravity_binned, full_info=True),
        BaselineSpec("ConditionalGravity", conditional_gravity.fit(train_loader, val_loader), full_info=True),
    ]

    results = [
        evaluate_baseline(
            spec.baseline,
            datamodule,
            loss,
            spec.name,
            cfg,
            wandb_params,
            full_info=spec.full_info,
        ).with_columns(seed=pl.lit(None, dtype=pl.Int64))
        for spec in baselines
    ]

    for train_seed in train_seeds:
        home_zone_mf = HomeZoneMFBaseline(num_nodes, is_home_idx, max_recall_k, seed=train_seed)
        results.append(
            evaluate_baseline(
                home_zone_mf.fit(train_loader, val_loader),
                datamodule,
                loss,
                "HomeZoneMF",
                cfg,
                wandb_params,
                run_tag=f"seed{train_seed}" if len(train_seeds) > 1 else None,
            ).with_columns(seed=pl.lit(train_seed, dtype=pl.Int64))
        )

    return results
