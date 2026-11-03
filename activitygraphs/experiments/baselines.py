"""Baseline runner: fit every baseline on the training split and evaluate it on the validation and test splits."""

from dataclasses import dataclass

import polars as pl

from activitygraphs.config import Config
from activitygraphs.datasets import SurveyDataModule
from activitygraphs.experiments.training import WandBParams, evaluate_baseline
from activitygraphs.models.baselines import (
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
from activitygraphs.models.scorers import FittedScorer
from activitygraphs.training.losses import Loss


@dataclass(frozen=True)
class BaselineSpec:
    """A baseline and associated reported name."""

    name: str
    baseline: FittedScorer


def closed_form_baselines(num_nodes: int) -> list[BaselineSpec]:
    """Return the specs of the baselines fitted once, rather than once per training seed.

    The baselines are returned unfitted.

    Returns:
        The frequency baselines, then the distance-decay ones, in fitting order. ``ConditionalGravity`` reads the
        ``Gravity`` fitted before it.
    """
    gravity = GravityBaseline()

    return [
        BaselineSpec("Uniform", UniformBaseline()),
        BaselineSpec("GlobalMarginal", GlobalBaseline()),
        BaselineSpec("VisitFrequency", VisitFrequencyBaseline(num_nodes)),
        BaselineSpec("ConditionalVisitFrequency", ConditionalVisitFrequencyBaseline(num_nodes)),
        BaselineSpec("DistanceDecay", DistanceDecayBaseline()),
        BaselineSpec("Gravity", gravity),
        BaselineSpec("GravityBinned", GravityBinnedBaseline()),
        BaselineSpec("ConditionalGravity", ConditionalGravityBaseline(gravity)),
    ]


def measure_baselines(
    num_nodes,
    datamodule: SurveyDataModule,
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
    train_users = datamodule.train_dataset.users
    val_users = datamodule.val_dataset.users
    locations = datamodule.locations
    results = []

    for spec in closed_form_baselines(num_nodes):
        baseline = spec.baseline.fit(train_users, val_users, locations)
        result = evaluate_baseline(baseline, datamodule, loss, spec.name, cfg, wandb_params)
        results.append(result.with_columns(seed=pl.lit(None, dtype=pl.Int64)))

    for train_seed in train_seeds:
        home_zone_mf = HomeZoneMFBaseline(num_nodes, max_recall_k, seed=train_seed)
        home_zone_mf_fitted = home_zone_mf.fit(train_users, val_users, locations)
        run_tag = f"seed{train_seed}" if len(train_seeds) > 1 else None

        home_zone_mf_results = evaluate_baseline(
            home_zone_mf_fitted, datamodule, loss, "HomeZoneMF", cfg, wandb_params, run_tag=run_tag
        ).with_columns(seed=pl.lit(train_seed, dtype=pl.Int64))

        results.append(home_zone_mf_results)

    return results
