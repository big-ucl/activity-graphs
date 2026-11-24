"""The experiment runner: trains the variants of an experiment and runs its restored variants, beside the baselines."""

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass

import lightning as L
import polars as pl
from lightning.pytorch.loggers import Logger

from activitygraphs.analysis import check_overfit_health
from activitygraphs.config import Config
from activitygraphs.datasets import SurveyDataModule
from activitygraphs.experiments.restore import RestoredVariant, restore_scorer
from activitygraphs.experiments.results import save_results
from activitygraphs.experiments.setup import ExperimentSetup, setup_experiment
from activitygraphs.experiments.training import evaluate_model, model_logger, train_model
from activitygraphs.experiments.variants import Variant, variant_hyperparams, with_overrides
from activitygraphs.models.architectures import build_scorer
from activitygraphs.models.scorers import Scorer
from activitygraphs.runs import LOGS_DIR

log = logging.getLogger(__name__)


def run_experiment(cfg: Config):
    """Train every variant and run every restored variant once per training seed, beside the baselines.

    The variants are listed in ``cfg.experiment.variants``, and the restored variants in ``cfg.transfer``.

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
    model_results = _run_variants(setup)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)

    if overfitting:
        _log_overfit_health(pl.concat(model_results, how="diagonal"), cfg.train.max_recall_k)


@dataclass(frozen=True)
class _VariantToTest:
    """A variant of an experiment, trained or restored, and ready to test.

    Attributes:
        scorer: The scorer of the variant, with the weights it is tested with.
        datamodule: The datamodule of the users it is tested on.
        fit_rows: One ``stage="fit"`` row per epoch of its training in this run. None when it was not trained in this
            run.
    """

    scorer: Scorer
    datamodule: SurveyDataModule
    fit_rows: list[dict] | None


def _run_variants(setup: ExperimentSetup) -> list[pl.DataFrame]:
    """Train every variant, then run every restored variant, once per training seed, and return the result frames.

    Only training is reseeded, so the seed spread across repeated runs of one variant measures
    training stochasticity alone rather than split variance. A restored variant reads the checkpoint of the same
    training seed in its recorded run.
    """
    cfg = setup.cfg
    loss = setup.loss
    models: list[Variant | RestoredVariant] = [*setup.variants, *setup.restored_variants]
    results = []

    for train_seed in setup.train_seeds:
        run_tag = f"seed{train_seed}" if len(setup.train_seeds) > 1 else None
        seed_column = pl.lit(train_seed, dtype=pl.Int64)

        for model in models:
            L.seed_everything(train_seed, workers=True)
            name = model.name
            hyperparams = _logged_hyperparams(model)

            with _model_logger(setup, name, hyperparams, run_tag) as logger:
                variant_to_test = _train_or_restore_variant(setup, model, train_seed, run_tag, logger)
                result_df = evaluate_model(
                    variant_to_test.scorer,
                    name,
                    variant_to_test.datamodule,
                    loss,
                    cfg,
                    logger,
                    variant_to_test.fit_rows,
                )

            results.append(result_df.with_columns(seed=seed_column, is_baseline=pl.lit(False)))

    return results


def _logged_hyperparams(model: Variant | RestoredVariant) -> dict:
    """Return the settings of ``model`` logged to W&B, with its ``model_type``: trained, restored or fine-tuned."""
    if isinstance(model, RestoredVariant):
        model_type = "fine_tuned" if model.fine_tune else "restored"
        return {"model_type": model_type, **model.hyperparams}

    return {"model_type": "trained", **variant_hyperparams(model)}


def _train_or_restore_variant(
    setup: ExperimentSetup,
    model: Variant | RestoredVariant,
    train_seed: int,
    run_tag: str | None,
    logger: Logger | None,
) -> _VariantToTest:
    """Train ``model`` from scratch, or restore it, and return it ready to test."""
    if isinstance(model, RestoredVariant):
        return _restore_variant(setup, model, train_seed, run_tag, logger)

    return _train_variant(setup, model, run_tag, logger)


def _train_variant(
    setup: ExperimentSetup, variant: Variant, run_tag: str | None, logger: Logger | None
) -> _VariantToTest:
    """Build ``variant`` from scratch, train it, and return it ready to test."""
    cfg = setup.cfg
    datamodule = setup.datamodule
    scorer = build_scorer(variant.model, variant.features, datamodule.locations, datamodule.schema, cfg.train.home_pe)
    run_name = _run_name(variant.name, run_tag)

    fit_rows = train_model(
        scorer, run_name, variant.model, datamodule, setup.loss, cfg, setup.run_dir, logger, setup.debug
    )

    return _VariantToTest(scorer, datamodule, fit_rows)


def _restore_variant(
    setup: ExperimentSetup,
    restored_variant: RestoredVariant,
    train_seed: int,
    run_tag: str | None,
    logger: Logger | None,
) -> _VariantToTest:
    """Restore ``restored_variant`` from its checkpoint, fine-tune it when asked, and return it ready to test."""
    cfg = setup.cfg
    datamodule = setup.scaled_datamodules[restored_variant.scaling]
    scorer = restore_scorer(restored_variant, train_seed, datamodule.locations, datamodule.schema)

    if not restored_variant.fine_tune:
        return _VariantToTest(scorer, datamodule, fit_rows=None)

    trained_model = restored_variant.variant.model
    fine_tune_lr = trained_model.lr * cfg.transfer.fine_tune_lr_scale
    fine_tune_model = with_overrides(trained_model, {"lr": fine_tune_lr})
    run_name = _run_name(restored_variant.name, run_tag)

    fit_rows = train_model(
        scorer, run_name, fine_tune_model, datamodule, setup.loss, cfg, setup.run_dir, logger, setup.debug
    )

    return _VariantToTest(scorer, datamodule, fit_rows)


def _model_logger(
    setup: ExperimentSetup, name: str, hyperparams: dict, run_tag: str | None
) -> AbstractContextManager[Logger | None]:
    """Open the logger of the run ``run_tag`` of the model ``name``, with ``hyperparams`` logged to W&B."""
    run_name = _run_name(name, run_tag)
    logged_hyperparams = {"model": name, "run_tag": run_tag, **hyperparams}
    log_dir = None if setup.run_dir is None else setup.run_dir / LOGS_DIR

    return model_logger(run_name, logged_hyperparams, setup.wandb_params, log_dir)


def _run_name(name: str, run_tag: str | None) -> str:
    """Return the name of the run ``run_tag`` of the model ``name``, in the logger and the checkpoint directory."""
    return name if run_tag is None else f"{name}-{run_tag}"


def _log_overfit_health(results: pl.DataFrame, max_recall_k: int) -> None:
    """Log the best ``train_avg_recall@K`` of each variant on the memorised batch."""
    with pl.Config(tbl_rows=-1, float_precision=4):
        health = str(check_overfit_health(results, max_recall_k))

    log.info("Overfit health: best train_avg_recall@%d on the memorised batch\n%s", max_recall_k, health)
