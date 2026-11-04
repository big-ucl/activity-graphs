"""Baseline runner: fit/train every baseline on the training split and evaluate it on the validation and test splits."""

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


def measure_baselines(
    datamodule: SurveyDataModule,
    loss: Loss,
    cfg: Config,
    wandb_params: WandBParams,
) -> list[pl.DataFrame]:
    """Fit and evaluate the closed-form baselines: the frequency baselines, then the distance-decay ones.

    Each baseline is fitted once, and its frames carry a null ``seed``.

    Returns:
        One result frame per baseline, in order.
    """
    datamodule.setup()
    train_users = datamodule.train_dataset.users
    val_users = datamodule.val_dataset.users
    locations = datamodule.locations
    results = []

    for spec in _closed_form_baselines(locations.num_nodes):
        baseline = spec.baseline.fit(train_users, val_users, locations)
        result = evaluate_baseline(baseline, datamodule, loss, spec.name, cfg, wandb_params)
        results.append(result.with_columns(seed=pl.lit(None, dtype=pl.Int64)))

    return results


def _closed_form_baselines(num_nodes: int) -> list[BaselineSpec]:
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
