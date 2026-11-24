"""Training orchestration: train_model, evaluate_model, evaluate_baseline, and their result frames."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import OmegaConf

from activitygraphs.base import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config, ModelConfig
from activitygraphs.datasets import SurveyDataModule
from activitygraphs.models.scorers import FittedScorer, Scorer
from activitygraphs.runs import CHECKPOINTS_DIR
from activitygraphs.training.callbacks import OverfitDebugCallback
from activitygraphs.training.lightning_module import RankingModule, load_scorer_weights
from activitygraphs.training.loggers import MetricsLogger
from activitygraphs.training.losses import Loss


def home_distance_frame(datamodule: SurveyDataModule) -> pl.DataFrame:
    """Distance from home to every node for each test user, as ``stage="test_distances"`` rows.

    The ``distances`` column is a list of one distance in metres per node, in node-index order. One frame per run: the
    distances are a property of the split, not of a model.
    """
    datamodule.setup()
    test_dataset = datamodule.test_dataset
    distances = datamodule.locations.distance_m[test_dataset.users.home_node]

    return pl.DataFrame(
        {"user_idx": test_dataset.user_idx.tolist(), "distances": distances.tolist()},
        schema_overrides={"distances": pl.List(pl.Float32)},
    ).with_columns(stage=pl.lit(HOME_DISTANCE_STAGE), epoch=pl.lit(None, dtype=pl.Int64))


@dataclass
class WandBParams:
    """W&B logging identity shared by every run of one experiment invocation.

    ``experiment`` is the name of the experiment that produced the run (``cfg.experiment.name``).
    It is carried into the group, the tags and the logged hyperparameters, so a run can be traced
    back to the experiment it belongs to from any of the three places W&B lets you filter on.

    Attributes:
        config: The resolved run config without ``models``. Logged as the hyperparameters of every W&B run.
        run_dir: The run directory, logged beside the config. None on a ``fast_dev_run``.
    """

    use_wandb: bool
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    dataset_name: str | None = None
    experiment: str | None = None
    config: dict = field(default_factory=dict)
    run_dir: str | None = None

    def tags(self, *extra: str) -> list[str]:
        """Run tags: the dataset, the experiment, and any run-specific extras, skipping the unset ones."""
        return [tag for tag in [self.dataset_name, self.experiment, *extra] if tag]


@contextmanager
def model_logger(
    run_name: str, hyperparams: dict, wandb_params: WandBParams, log_dir: Path | None
) -> Iterator[Logger | None]:
    """Open the logger of one model: a W&B run when ``wandb_params.use_wandb`` is set, else a CSV log.

    Args:
        run_name: Name of the logger run.
        hyperparams: The settings of the model, logged to W&B beside ``wandb_params.config``. Its ``model_type``
            is added to the W&B tags.
        wandb_params: parameters to configure WandB logging.
        log_dir: Directory of the logs. None writes no CSV log.

    Yields:
        The logger, or None when W&B is off and ``log_dir`` is None. The W&B run is finished on exit.
    """
    save_dir = "." if log_dir is None else str(log_dir)

    if not wandb_params.use_wandb:
        csv_logger = None if log_dir is None else CSVLogger(save_dir=save_dir, name=run_name)
        yield csv_logger
        return

    tags = wandb_params.tags(hyperparams["model_type"])
    logger = WandbLogger(
        project=wandb_params.project,
        entity=wandb_params.entity,
        name=run_name,
        group=wandb_params.group,
        save_dir=save_dir,
        tags=tags,
    )
    logged_params = {**wandb_params.config, "run_dir": wandb_params.run_dir, **hyperparams}
    logger.log_hyperparams(logged_params)

    try:
        yield logger
    finally:
        import wandb

        wandb.finish()


def evaluate_baseline(
    baseline: FittedScorer,
    datamodule: SurveyDataModule,
    loss: Loss,
    name: str,
    cfg: Config,
    wandb_params: WandBParams,
) -> pl.DataFrame:
    """Evaluate a fitted baseline and return its result frame, in the layout of ``evaluate_model``.

    Args:
        baseline: Fitted baseline module to evaluate.
        datamodule: ``SurveyDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance, for the BCE and the monitored metric.
        name: Baseline name used for logging and in the returned frame.
        cfg: The config used by the experiment.
        wandb_params: parameters to configure WandB logging.
    """
    hyperparams = {"model": name, "run_tag": None, "model_type": "baseline"}

    with model_logger(name, hyperparams, wandb_params, log_dir=None) as logger:
        results = evaluate_model(baseline, name, datamodule, loss, cfg, logger, fit_rows=None)

    return results.with_columns(is_baseline=pl.lit(True))


def train_model(
    scorer: Scorer,
    run_name: str,
    model: ModelConfig,
    datamodule: SurveyDataModule,
    loss: Loss,
    cfg: Config,
    run_dir: Path | None,
    logger: Logger | None,
    debug: bool,
) -> list[dict]:
    """Train ``scorer`` on the training users, leave in it the weights of its best epoch, and return the epoch rows.

    Args:
        scorer: Scorer to train, built from scratch or restored from a checkpoint.
        run_name: Name of the training run, which names its checkpoint directory.
        model: The model config, giving the learning rate and the weight decay.
        datamodule: ``SurveyDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance
        cfg: The config used by the experiment. ``cfg.trainer`` holds the ``Trainer`` arguments, and ``cfg.train``
            the training settings.
        run_dir: Run directory. Checkpoints are written to ``checkpoints/<run_name>/`` (``best.ckpt``,
            ``last.ckpt``). None writes no checkpoint and leaves the weights of the last epoch in ``scorer``.
        logger: The logger of the model, from ``model_logger``. None keeps the metrics in memory only.
        debug: Flag that enables `OverfitDebugCallback` statistics printing at the start and end of training.

    Returns:
        One ``stage="fit"`` row per epoch, with the ``epoch``, the ``train_loss`` and the ``val_*`` metrics.
    """
    train = cfg.train

    datamodule.setup()

    lr_scheduler = cast(dict, OmegaConf.to_container(train.lr_scheduler)) if train.schedule_lr else None
    recall_ks = list(cfg.analysis.recall_curve_ks)

    lit_model = RankingModule(
        scorer=scorer,
        locations=datamodule.locations,
        lr=model.lr,
        loss=loss,
        max_recall_k=train.max_recall_k,
        recall_ks=recall_ks,
        weight_decay=model.weight_decay,
        lr_scheduler=lr_scheduler,
        calibration_bins=train.calibration_bins,
        compile_model=train.compile,
        store_score_vectors=train.save_score_vectors,
        log_train_ranking=train.log_train_ranking,
    )

    callbacks = _build_callbacks(loss, cfg, run_dir, run_name, debug)
    metrics_logger = MetricsLogger()
    loggers: list[Logger] = [metrics_logger] if logger is None else [logger, metrics_logger]

    trainer_arguments = _trainer_arguments(cfg)
    trainer = L.Trainer(**trainer_arguments, logger=loggers, callbacks=callbacks)
    trainer.fit(lit_model, datamodule=datamodule)

    if run_dir is not None:
        best_checkpoint = cast(ModelCheckpoint, trainer.checkpoint_callback)
        best_path = Path(best_checkpoint.best_model_path)
        load_scorer_weights(scorer, best_path)

    return metrics_logger.rows


def evaluate_model(
    scorer: Scorer,
    name: str,
    datamodule: SurveyDataModule,
    loss: Loss,
    cfg: Config,
    logger: Logger | None,
    fit_rows: list[dict] | None,
) -> pl.DataFrame:
    """Test ``scorer`` with its weights as they are, and return its result frame.

    Args:
        scorer: The scorer tested. It is not trained here.
        name: Name of the model in the returned frame.
        datamodule: ``SurveyDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance, for the BCE and the monitored metric.
        cfg: The config used by the experiment. ``cfg.trainer`` holds the ``Trainer`` arguments, ``cfg.train`` the
            evaluation settings, and ``analysis.recall_curve_ks`` the ``recall@k`` cutoffs.
        logger: The logger of the model, from ``model_logger``. None logs nothing.
        fit_rows: The epoch rows returned by ``train_model`` when ``scorer`` was trained. None validates ``scorer``
            once, as one ``stage="fit"`` row at epoch 0.

    Returns:
        Long-format DataFrame tagged by ``stage``: the ``stage="fit"`` rows, one ``stage="test"`` row carrying the
        ``test_*`` metrics, one ``stage="test_user"`` row per test user, and one ``stage="test_scores"`` row per test
        user when the score vectors are stored. Cells absent for a row's stage are null.
    """
    datamodule.setup()
    recall_ks = list(cfg.analysis.recall_curve_ks)

    module = RankingModule(
        scorer=scorer,
        locations=datamodule.locations,
        lr=0.0,
        loss=loss,
        max_recall_k=cfg.train.max_recall_k,
        recall_ks=recall_ks,
        calibration_bins=cfg.train.calibration_bins,
        store_score_vectors=cfg.train.save_score_vectors,
    )

    trainer_logger = False if logger is None else logger
    trainer_arguments = _trainer_arguments(cfg)
    trainer = L.Trainer(**trainer_arguments, logger=trainer_logger, enable_autolog_hparams=False)

    if fit_rows is None:
        (val_results,) = trainer.validate(module, datamodule=datamodule)
        fit_rows = [{"stage": "fit", "epoch": 0, "train_loss": 0.0, **val_results}]

    (test_results,) = trainer.test(module, datamodule=datamodule)
    test_row = {"stage": "test", "epoch": None, **test_results}

    aggregate_rows = [*fit_rows, test_row]
    aggregate_results = _aggregate_frame(aggregate_rows, name)
    per_user_results = _per_user_frame(module, name, datamodule.home_coverage)
    score_vectors = _score_vector_frame(module, name)

    results = pl.concat([aggregate_results, per_user_results, score_vectors], how="diagonal")
    first_cols = ["name", "stage", "epoch"]
    other_cols = [c for c in results.columns if c not in first_cols]

    return results.select(first_cols + other_cols)


def _score_vector_frame(module: RankingModule, name: str) -> pl.DataFrame:
    """Full per-node test score vectors as ``stage="test_scores"`` rows, one row per scored test user.

    The ``scores`` column is a list of one score per node, in node-index order, row-aligned with the
    ``test_user`` rows of the same run. Empty when the module was built without ``store_score_vectors``.
    """
    if not module.per_user_score_vectors:
        return pl.DataFrame()

    return pl.DataFrame(
        {
            "user_idx": module.per_user_columns["user_idx"],
            "scores": module.per_user_score_vectors,
        },
        schema_overrides={"scores": pl.List(pl.Float32)},
    ).with_columns(
        name=pl.lit(name),
        stage=pl.lit(SCORE_VECTOR_STAGE),
        epoch=pl.lit(None, dtype=pl.Int64),
    )


def _aggregate_frame(rows: list[dict], name: str) -> pl.DataFrame:
    """Collected ``stage="fit"``/``stage="test"`` rows as one named frame.

    Scans every row for the schema: the single test row comes last, after one fit row per epoch, so
    Polars' default 100-row inference window would silently drop the ``test_*`` metrics on any run
    longer than that many epochs.
    """
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(name=pl.lit(name))


def _per_user_frame(module: RankingModule, name: str, home_coverage_df: pl.DataFrame) -> pl.DataFrame:
    """Per-user test scores as ``stage="test_user"`` rows, one row per scored test user.

    Carried inside the same long-format frame as the aggregate rows so that every caller keeps a
    single return value; ``save_results`` splits the two stages into separate parquet files.
    """
    if not module.per_user_columns:
        return pl.DataFrame()

    per_user_df = pl.DataFrame(module.per_user_columns).with_columns(
        name=pl.lit(name),
        stage=pl.lit(PER_USER_STAGE),
        epoch=pl.lit(None, dtype=pl.Int64),
    )

    return per_user_df.join(home_coverage_df, on="user_idx", how="left")


def _build_callbacks(loss: Loss, cfg: Config, run_dir: Path | None, run_name: str, debug: bool) -> list[L.Callback]:
    """Build the ``Trainer`` callbacks of one training run.

    Args:
        loss: ``Loss`` instance, whose ``monitor`` metric and ``monitor_mode`` direction the checkpoint and early
            stopping follow.
        cfg: The config used by the experiment. ``cfg.train.early_stopping.patience`` sets the early stopping;
            ``cfg.trainer.overfit_batches > 0`` turns it off.
        run_dir: Run directory. Checkpoints are written to ``checkpoints/<run_name>/``. None disables them.
        run_name: Name of the training run, which names its checkpoint directory.
        debug: Flag that adds the ``OverfitDebugCallback``.

    Returns:
        The best and the last checkpoint when ``run_dir`` is set, early stopping unless its patience is None or the run
        is overfitting, and the ``OverfitDebugCallback`` when ``debug`` is set.
        ``best.ckpt`` holds the epoch with the best monitored metric, and ``last.ckpt`` the latest epoch. The best
        checkpoint comes first, since ``trainer.checkpoint_callback`` is the first checkpoint callback.
    """
    callbacks: list[L.Callback] = []

    if run_dir is not None:
        checkpoint_dir = str(run_dir / CHECKPOINTS_DIR / run_name)
        best = ModelCheckpoint(
            dirpath=checkpoint_dir, filename="best", save_top_k=1, monitor=loss.monitor, mode=loss.monitor_mode
        )
        last = ModelCheckpoint(dirpath=checkpoint_dir, filename="last", save_top_k=1, monitor=None)
        callbacks += [best, last]

    patience = cfg.train.early_stopping.patience
    if patience is not None and cfg.trainer.overfit_batches == 0:
        callbacks.append(EarlyStopping(monitor=loss.monitor, mode=loss.monitor_mode, patience=patience))

    if debug:
        callbacks.append(OverfitDebugCallback())

    return callbacks


def _trainer_arguments(cfg: Config) -> dict:
    """Return the keyword arguments of the Lightning ``Trainer`` from ``cfg.trainer``, with interpolations resolved."""
    return cast(dict, OmegaConf.to_container(cfg.trainer, resolve=True))
