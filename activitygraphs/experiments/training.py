"""Training orchestration: train_and_evaluate_model, evaluate_baseline, and their result frames."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from activitygraphs.base import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config
from activitygraphs.datasets import VisitDataModule
from activitygraphs.experiments.variants import Variant, variant_hyperparams
from activitygraphs.ml.baselines.gravity import unscale_distances
from activitygraphs.ml.callbacks import OverfitDebugCallback
from activitygraphs.ml.lightning_module import RankingModule
from activitygraphs.ml.loggers import MetricsLogger
from activitygraphs.ml.losses import Loss
from activitygraphs.runs import CHECKPOINTS_DIR, LOGS_DIR


def home_distance_frame(datamodule: VisitDataModule) -> pl.DataFrame:
    """Distance from home to every node for each test user, as ``stage="test_distances"`` rows.

    The ``distances`` column is a list of one distance in metres per node, in node-index order. One frame per run: the
    distances are a property of the split, not of a model.
    """
    datamodule.setup()
    test_dataset = datamodule.test_dataset
    user_idx = torch.as_tensor(list(test_dataset.indices()), dtype=torch.long)
    distances = unscale_distances(test_dataset.distances[user_idx].squeeze(-1), datamodule.scalers.distances)

    return pl.DataFrame(
        {"user_id": user_idx.tolist(), "distances": distances.tolist()},
        schema_overrides={"distances": pl.List(pl.Float32)},
    ).with_columns(stage=pl.lit(HOME_DISTANCE_STAGE), epoch=pl.lit(None, dtype=pl.Int64))


def score_vector_frame(module: RankingModule, name: str) -> pl.DataFrame:
    """Full per-node test score vectors as ``stage="test_scores"`` rows, one row per scored test user.

    The ``scores`` column is a list of one score per node, in node-index order, row-aligned with the
    ``test_user`` rows of the same run. Empty when the module was built without ``store_score_vectors``.
    """
    if not module.per_user_score_vectors:
        return pl.DataFrame()

    return pl.DataFrame(
        {
            "user_id": module.per_user_columns["user_id"],
            "scores": module.per_user_score_vectors,
        },
        schema_overrides={"scores": pl.List(pl.Float32)},
    ).with_columns(
        name=pl.lit(name),
        stage=pl.lit(SCORE_VECTOR_STAGE),
        epoch=pl.lit(None, dtype=pl.Int64),
    )


def aggregate_frame(rows: list[dict], name: str) -> pl.DataFrame:
    """Collected ``stage="fit"``/``stage="test"`` rows as one named frame.

    Scans every row for the schema: the single test row comes last, after one fit row per epoch, so
    Polars' default 100-row inference window would silently drop the ``test_*`` metrics on any run
    longer than that many epochs.
    """
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(name=pl.lit(name))


def per_user_frame(module: RankingModule, name: str, home_coverage_df: pl.DataFrame) -> pl.DataFrame:
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

    return per_user_df.join(home_coverage_df, on="user_id", how="left")


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


def evaluate_baseline(
    baseline: torch.nn.Module,
    datamodule: VisitDataModule,
    loss: Loss,
    name: str,
    cfg: Config,
    wandb_params: WandBParams,
    full_info: bool = False,
    run_tag: str | None = None,
) -> pl.DataFrame:
    """Evaluate a baseline model and return a results DataFrame matching the ``run_experiment`` format.

    Returns two stage-tagged rows (one ``stage="fit"`` with the ``val_*`` metrics, one
    ``stage="test"`` with the ``test_*`` metrics) so baseline frames align with the
    ``run_experiment`` output.

    Args:
        baseline: Fitted baseline module to evaluate.
        datamodule: ``VisitDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance, for the BCE and the monitored metric.
        name: Baseline name used for logging and in the returned frame.
        cfg: The config used by the experiment, read for the evaluation settings under ``train`` and the
            ``recall@k`` cutoffs under ``analysis``.
        wandb_params: parameters to configure WandB logging.
        full_info: Append each node's distance from home as the last feature column, for the distance baselines.
        run_tag: Suffix distinguishing repeated fits of one baseline in the logger, e.g. per training seed. It
            does not reach the returned frame, whose ``name`` stays ``name``.
    """
    baseline_module = RankingModule(
        model=baseline,
        lr=0.0,
        loss=loss,
        max_recall_k=cfg.train.max_recall_k,
        recall_ks=list(cfg.analysis.recall_curve_ks),
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        calibration_bins=cfg.train.calibration_bins,
        store_score_vectors=cfg.train.save_score_vectors,
        full_info=full_info,
    )

    if wandb_params.use_wandb:
        logger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name if run_tag is None else f"{name}-{run_tag}",
            group=wandb_params.group,
            tags=wandb_params.tags("baseline"),
        )
        logger.log_hyperparams({
            **wandb_params.config,
            "run_dir": wandb_params.run_dir,
            "model": name,
            "run_tag": run_tag,
            "model_type": "baseline",
        })

    else:
        logger = False

    trainer = L.Trainer(logger=logger, enable_progress_bar=False)

    try:
        (val_results,) = trainer.validate(baseline_module, datamodule=datamodule)
        (test_results,) = trainer.test(baseline_module, datamodule=datamodule)
    finally:
        if wandb_params.use_wandb:
            import wandb

            wandb.finish()

    fit_row = {"name": name, "stage": "fit", "epoch": 0, "train_loss": 0.0, **val_results}
    test_row = {"name": name, "stage": "test", "epoch": None, **test_results}

    aggregate_results = pl.DataFrame([fit_row, test_row])
    per_user_results = per_user_frame(baseline_module, name, datamodule.home_coverage)
    score_vectors = score_vector_frame(baseline_module, name)

    return pl.concat([aggregate_results, per_user_results, score_vectors], how="diagonal").with_columns(
        is_baseline=pl.lit(True)
    )


def train_and_evaluate_model(
    model: torch.nn.Module,
    variant: Variant,
    datamodule: VisitDataModule,
    loss: Loss,
    cfg: Config,
    run_dir: Path | None,
    wandb_params: WandBParams,
    debug: bool,
    run_tag: str | None = None,
) -> pl.DataFrame:
    """Train a model and return per-epoch metrics as a Polars DataFrame.

    Args:
        model: Model to train, must implement ``forward(x, edge_index, edge_attr, batch)``.
        variant: The variant ``model`` was built from, giving its name, optimiser settings and input features.
        datamodule: ``VisitDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance
        cfg: The config used by the experiment. ``cfg.trainer`` holds the ``Trainer`` arguments, ``cfg.train`` the
            training and evaluation settings, and ``analysis.recall_curve_ks`` the ``recall@k`` cutoffs.
        run_dir: Run directory. Checkpoints are written to ``checkpoints/<run name>/`` (``best.ckpt``,
            ``last.ckpt``) and the logs to ``logs/``. None disables both.
        wandb_params: parameters to configure WandB logging.
        debug: Flag that enables `OverfitDebugCallback` statistics printing at the start and end of training.
        run_tag: Suffix distinguishing repeated runs of the same model (e.g. per training seed). It
            qualifies the checkpoint directory and logger run name only; the variant name still identifies
            the model in the returned frame, so repeated runs group together.

    Returns:
        Long-format DataFrame tagged by ``stage``: one ``stage="fit"`` row per epoch carrying
        ``train_loss`` and the ``val_*`` metrics, plus (when a test run executed) one
        ``stage="test"`` row carrying the ``test_*`` metrics and one ``stage="test_user"`` row per
        test user carrying that user's ``r_precision``. Columns: ``name``, ``stage``,
        ``epoch``, ``train_loss``, ``val_*``, ``test_*``, ``user_id``, ``n_pos``, ``r_precision``;
        cells absent for a row's stage are null. Empty on a ``fast_dev_run``.
    """
    name = variant.name
    run_name = name if run_tag is None else f"{name}-{run_tag}"
    train = cfg.train

    datamodule.setup()

    lit_model = RankingModule(
        model=model,
        lr=variant.model.lr,
        loss=loss,
        full_info=variant.full_info,
        use_demographics=variant.use_demographics,
        max_recall_k=train.max_recall_k,
        recall_ks=list(cfg.analysis.recall_curve_ks),
        weight_decay=variant.model.weight_decay,
        lr_scheduler=cast(dict, OmegaConf.to_container(train.lr_scheduler)) if train.schedule_lr else None,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        use_home_pe=variant.use_home_pe,
        home_pe_bins=train.home_pe.bins,
        home_pe_max_hops=train.home_pe.max_hops,
        calibration_bins=train.calibration_bins,
        compile_model=train.compile,
        store_score_vectors=train.save_score_vectors,
        log_train_ranking=train.log_train_ranking,
    )

    # Build the callbacks

    callbacks: list[L.Callback] = [LearningRateMonitor(logging_interval="epoch")]

    if run_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir / CHECKPOINTS_DIR / run_name),
                filename="best",
                save_last=True,
                save_top_k=1,
                monitor=loss.monitor,
                mode=loss.monitor_mode,
            )
        )

    if debug:
        callbacks.append(OverfitDebugCallback())

    log_dir = str(run_dir / LOGS_DIR) if run_dir is not None else "."

    # Configure logging

    if wandb_params.use_wandb:
        logger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=run_name,
            group=wandb_params.group,
            save_dir=log_dir,
            tags=wandb_params.tags(),
        )
        logger.log_hyperparams({
            **wandb_params.config,
            "run_dir": wandb_params.run_dir,
            "model": name,
            "run_tag": run_tag,
            **variant_hyperparams(variant),
        })
    else:
        logger = CSVLogger(save_dir=log_dir, name=run_name)

    metrics_logger = MetricsLogger()

    # Create trainer and fit

    trainer = L.Trainer(
        **cast(dict, OmegaConf.to_container(cfg.trainer)), logger=[logger, metrics_logger], callbacks=callbacks
    )

    trainer.fit(lit_model, datamodule=datamodule)

    # Process results

    try:
        if cfg.trainer.fast_dev_run:
            return pl.DataFrame()

        if run_dir is not None:
            trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

        aggregate_results = aggregate_frame(metrics_logger.rows, name)
        per_user_results = per_user_frame(lit_model, name, datamodule.home_coverage)
        score_vectors = score_vector_frame(lit_model, name)

        results = pl.concat([aggregate_results, per_user_results, score_vectors], how="diagonal").with_columns(
            is_baseline=pl.lit(False)
        )

        first_cols = ["name", "stage", "epoch"]
        other_cols = [c for c in results.columns if c not in first_cols]
        results = results.select(first_cols + other_cols)

        return results
    finally:
        if wandb_params.use_wandb:
            import wandb

            wandb.finish()
