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
    LogsumGravityBaseline,
    TravelTimeGravityBaseline,
    UniformBaseline,
    VisitFrequencyBaseline,
)
from activitygraphs.models.scorers import FittedScorer
from activitygraphs.schema import TensorSchema
from activitygraphs.training.losses import Loss


@dataclass(frozen=True)
class BaselineSpec:
    """A baseline and associated reported name.

    Attributes:
        name: The reported name.
        baseline: The baseline, unfitted.
        reads_visits: Whether fitting the baseline reads the visits of the training users.
    """

    name: str
    baseline: FittedScorer
    reads_visits: bool = True


def measure_baselines(
    datamodule: SurveyDataModule,
    loss: Loss,
    cfg: Config,
    wandb_params: WandBParams,
    fit_datamodule: SurveyDataModule | None = None,
    name_suffix: str = "",
    visit_free_only: bool = False,
) -> list[pl.DataFrame]:
    """Fit and evaluate the closed-form baselines: the frequency baselines, then the distance-decay ones.

    Each baseline is fitted once, and its frames carry a null ``seed``.

    Args:
        datamodule: The datamodule of the users the baselines are `evaluated` on.
        loss: ``Loss`` instance, used for training and validation.
        cfg: The config used by the experiment.
        wandb_params: parameters to configure WandB logging.
        fit_datamodule: The datamodule of the users the baselines are `fitted` on, such as another survey over the same
            nodes. Fits on ``datamodule`` if ``None``.
        name_suffix: Suffix appended to the name of every baseline.
        visit_free_only: Fit and evaluate only the baselines that do not read visits.

    Returns:
        One result frame per baseline, in order.

    Raises:
        ValueError: If ``fit_datamodule`` has other nodes than ``datamodule``, in another order.
    """
    datamodule.setup()
    fit_datamodule = datamodule if fit_datamodule is None else fit_datamodule
    fit_datamodule.setup()

    locations = fit_datamodule.locations
    if not locations.node_ids.equals(datamodule.locations.node_ids):
        raise ValueError("Baselines must be fit on the same nodes they are scored on.")

    train_users = fit_datamodule.train_dataset.users
    val_users = fit_datamodule.val_dataset.users
    travel_times = fit_datamodule.schema.travel_times
    results = []

    for spec in _closed_form_baselines(locations.num_nodes, travel_times):
        if visit_free_only and spec.reads_visits:
            continue

        name = f"{spec.name}{name_suffix}"

        baseline = spec.baseline.fit(train_users, val_users, locations)
        result = evaluate_baseline(baseline, datamodule, loss, name, cfg, wandb_params)

        results.append(result.with_columns(seed=pl.lit(None, dtype=pl.Int64)))

    return results


def _closed_form_baselines(num_nodes: int, travel_time_schema: TensorSchema) -> list[BaselineSpec]:
    """Return the specs of the baselines fitted once, rather than once per training seed.

    The baselines are returned unfitted.

    Args:
        num_nodes: Number of nodes of the location graph.
        travel_time_schema: The column schema of the travel times, read by ``GravityTT``, ``GravityLogsum`` and
            ``GravityLogsumDist``.

    Returns:
        The frequency baselines, then the distance-decay ones, in fitting order. ``ConditionalGravity`` reads the
        ``Gravity`` fitted before it.
    """
    gravity = GravityBaseline()

    return [
        BaselineSpec("Uniform", UniformBaseline(), reads_visits=False),
        BaselineSpec("GlobalMarginal", GlobalBaseline()),
        BaselineSpec("VisitFrequency", VisitFrequencyBaseline(num_nodes)),
        BaselineSpec("ConditionalVisitFrequency", ConditionalVisitFrequencyBaseline(num_nodes)),
        BaselineSpec("DistanceDecay", DistanceDecayBaseline(), reads_visits=False),
        BaselineSpec("Gravity", gravity),
        BaselineSpec("GravityTT", TravelTimeGravityBaseline(travel_time_schema)),
        BaselineSpec("GravityLogsum", LogsumGravityBaseline(travel_time_schema, with_distance=False)),
        BaselineSpec("GravityLogsumDist", LogsumGravityBaseline(travel_time_schema, with_distance=True)),
        BaselineSpec("GravityBinned", GravityBinnedBaseline()),
        BaselineSpec("ConditionalGravity", ConditionalGravityBaseline(gravity)),
    ]
