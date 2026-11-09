"""The experiment runner: trains the variant list of ``cfg.experiment`` beside the baselines."""

import logging

import lightning as L
import polars as pl

from activitygraphs.analysis import check_overfit_health
from activitygraphs.config import Config
from activitygraphs.experiments.results import save_results
from activitygraphs.experiments.setup import ExperimentSetup, setup_experiment
from activitygraphs.experiments.training import train_and_evaluate_model
from activitygraphs.experiments.variants import Variant, build_variants
from activitygraphs.models.architectures import build_scorer

log = logging.getLogger(__name__)


def run_experiment(cfg: Config):
    """Train every variant of ``cfg.experiment.variants`` once per training seed, beside the baselines.

    Results are logged to W&B and written to ``results.parquet`` in the run directory, with the per-user test scores
    alongside in ``per-user.parquet``. The baselines are deterministic, so they are evaluated once.

    While overfitting (``cfg.trainer.overfit_batches > 0``), the baselines are skipped, and the best
    ``train_avg_recall@K`` of each variant on the memorised batch is logged. A healthy model drives it to ~1.0.

    Raises:
        ValueError: If the run overfits with ``train.log_train_ranking`` off, since the health check reads
            ``train_avg_recall@K``.
    """
    overfitting = cfg.trainer.overfit_batches > 0
    if overfitting and not cfg.train.log_train_ranking:
        raise ValueError("An overfitting run needs train.log_train_ranking=true.")

    setup = setup_experiment(cfg, with_baselines=not overfitting)
    variants = build_variants(cfg.experiment.variants, setup.models)

    model_results = _run_variants(setup, variants)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)

    if overfitting:
        _log_overfit_health(pl.concat(model_results, how="diagonal"), cfg.train.max_recall_k)


def _run_variants(setup: ExperimentSetup, variants: list[Variant]) -> list[pl.DataFrame]:
    """Train every variant once per training seed and return the seed-tagged result frames.

    Only training is reseeded, so the seed spread across repeated runs of one variant measures
    training stochasticity alone rather than split variance.
    """
    cfg = setup.cfg
    locations = setup.datamodule.locations
    schema = setup.datamodule.schema
    results = []

    for train_seed in setup.train_seeds:
        for variant in variants:
            L.seed_everything(train_seed, workers=True)
            scorer = build_scorer(variant.model, variant.features, locations, schema, cfg.train.home_pe)
            run_tag = f"seed{train_seed}" if len(setup.train_seeds) > 1 else None

            result_df = train_and_evaluate_model(
                scorer=scorer,
                variant=variant,
                datamodule=setup.datamodule,
                loss=setup.loss,
                cfg=cfg,
                run_dir=setup.run_dir,
                wandb_params=setup.wandb_params,
                debug=setup.debug,
                run_tag=run_tag,
            )
            results.append(result_df.with_columns(seed=pl.lit(train_seed, dtype=pl.Int64)))

    return results


def _log_overfit_health(results: pl.DataFrame, max_recall_k: int) -> None:
    """Log the best ``train_avg_recall@K`` of each variant on the memorised batch."""
    with pl.Config(tbl_rows=-1, float_precision=4):
        health = str(check_overfit_health(results, max_recall_k))

    log.info("Overfit health: best train_avg_recall@%d on the memorised batch\n%s", max_recall_k, health)
