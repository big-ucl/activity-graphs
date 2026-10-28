"""Baseline runner: fit every baseline on the training split and evaluate it on the validation and test splits."""

from dataclasses import dataclass

import polars as pl

from activitygraphs.config import Config
from activitygraphs.datasets import VisitDataModule
from activitygraphs.experiments.training import WandBParams, evaluate_baseline
from activitygraphs.ml.baselines import (
    Baseline,
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
    """A fitted baseline and associated reported name."""

    name: str
    baseline: Baseline


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
    is_home_idx_user_node = datamodule.train_dataset.schema.user_nodes.index_of("is_home")

    gravity = GravityBaseline(is_home_idx_user_node).fit(train_loader)
    gravity_binned = GravityBinnedBaseline(is_home_idx_user_node).fit(train_loader)
    conditional_visit_frequency = ConditionalVisitFrequencyBaseline(num_nodes, is_home_idx_user_node)
    conditional_gravity = ConditionalGravityBaseline(gravity)

    baselines = [
        BaselineSpec("Uniform", UniformBaseline()),
        BaselineSpec("GlobalMarginal", GlobalBaseline().fit(train_loader)),
        BaselineSpec("VisitFrequency", VisitFrequencyBaseline(num_nodes, is_home_idx_user_node).fit(train_loader)),
        BaselineSpec("ConditionalVisitFrequency", conditional_visit_frequency.fit(train_loader, val_loader)),
        BaselineSpec("DistanceDecay", DistanceDecayBaseline()),
        BaselineSpec("Gravity", gravity),
        BaselineSpec("GravityBinned", gravity_binned),
        BaselineSpec("ConditionalGravity", conditional_gravity.fit(train_loader, val_loader)),
    ]

    results = [
        evaluate_baseline(spec.baseline, datamodule, loss, spec.name, cfg, wandb_params).with_columns(
            seed=pl.lit(None, dtype=pl.Int64)
        )
        for spec in baselines
    ]

    for train_seed in train_seeds:
        home_zone_mf = HomeZoneMFBaseline(num_nodes, is_home_idx_user_node, max_recall_k, seed=train_seed)
        home_zone_mf_fitted = home_zone_mf.fit(train_loader, val_loader)
        run_tag = f"seed{train_seed}" if len(train_seeds) > 1 else None

        home_zone_mf_results = evaluate_baseline(
            home_zone_mf_fitted, datamodule, loss, "HomeZoneMF", cfg, wandb_params, run_tag=run_tag
        ).with_columns(seed=pl.lit(train_seed, dtype=pl.Int64))

        results.append(home_zone_mf_results)

    return results
