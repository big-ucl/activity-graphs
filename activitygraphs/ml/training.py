"""Training orchestration: run_experiment, evaluate_baseline, and shared feature/weight helpers."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from activitygraphs.ml.baselines.gravity import unscale_distances
from activitygraphs.ml.callbacks import EpochMetricsCollector, OverfitDebugCallback
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.losses import Loss

PER_USER_STAGE = "test_user"
SCORE_VECTOR_STAGE = "test_scores"
HOME_DISTANCE_STAGE = "test_distances"


def home_distance_frame(datamodule: ActivityDataModule) -> pl.DataFrame:
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


def score_vector_frame(module: ActivityGraphModule, name: str) -> pl.DataFrame:
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


def per_user_frame(module: ActivityGraphModule, name: str, home_coverage_df: pl.DataFrame) -> pl.DataFrame:
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

    ``experiment`` is the name of the experiment that produced the run (``cfg.train.experiment``).
    It is carried into the group, the tags and the logged hyperparameters, so a run can be traced
    back to the experiment it belongs to from any of the three places W&B lets you filter on.
    """

    use_wandb: bool
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    dataset_name: str | None = None
    experiment: str | None = None

    def tags(self, *extra: str) -> list[str]:
        """Run tags: the dataset, the experiment, and any run-specific extras, skipping the unset ones."""
        return [tag for tag in [self.dataset_name, self.experiment, *extra] if tag]


def evaluate_baseline(
    baseline: torch.nn.Module,
    datamodule: ActivityDataModule,
    loss: Loss,
    name: str,
    max_recall_k: int,
    recall_ks: Sequence[int],
    wandb_params: WandBParams | None = None,
    store_score_vectors: bool = False,
    full_info: bool = False,
    run_tag: str | None = None,
) -> pl.DataFrame:
    """Evaluate a baseline model and return a results DataFrame matching the ``run_experiment`` format.

    Returns two stage-tagged rows (one ``stage="fit"`` with the ``val_*`` metrics, one
    ``stage="test"`` with the ``test_*`` metrics) so baseline frames align with the
    ``run_experiment`` output.

    Args:
        full_info: Append each node's distance from home as the last feature column, for the distance baselines.
        run_tag: Suffix distinguishing repeated fits of one baseline in the logger, e.g. per training seed. It
            does not reach the returned frame, whose ``name`` stays ``name``.
    """
    baseline_module = ActivityGraphModule(
        model=baseline,
        lr=0.0,
        loss=loss,
        max_recall_k=max_recall_k,
        recall_ks=recall_ks,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        store_score_vectors=store_score_vectors,
        full_info=full_info,
    )

    if wandb_params and wandb_params.use_wandb:
        logger: WandbLogger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name if run_tag is None else f"{name}-{run_tag}",
            group=wandb_params.group,
            tags=wandb_params.tags("baseline"),
        )
        logger.log_hyperparams({
            "model": name,
            "run_tag": run_tag,
            "dataset": wandb_params.dataset_name,
            "experiment": wandb_params.experiment,
            "model_type": "baseline",
        })

    else:
        logger = False

    trainer = L.Trainer(logger=logger, enable_progress_bar=False)

    try:
        (val_results,) = trainer.validate(baseline_module, datamodule=datamodule)
        (test_results,) = trainer.test(baseline_module, datamodule=datamodule)
    finally:
        if wandb_params and wandb_params.use_wandb:
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
    datamodule: ActivityDataModule,
    loss: Loss,
    max_recall_k: int,
    recall_ks: Sequence[int],
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    reg: str | None = None,
    full_info: bool = False,
    use_demographics: bool = True,
    model_save_dir: Path | None = None,
    fast_dev_run: bool = False,
    overfit_batches: int = 0,
    weight_decay: float = 1e-4,
    schedule_lr: bool = False,
    pop_mode: Literal["none", "offset", "feature"] = "none",
    use_home_pe: bool = False,
    store_score_vectors: bool = False,
    log_train_ranking: bool = True,
    extra_hyperparams: dict | None = None,
    compile_model: bool = True,
    wandb_params: WandBParams | None = None,
    debug: bool = False,
    run_tag: str | None = None,
) -> pl.DataFrame:
    """Train a model and return per-epoch metrics as a Polars DataFrame.

    Uses AdamW with weight decay 1e-4 and ReduceLROnPlateau scheduling.

    Args:
        model: Model to train, must implement ``forward(x, edge_index, edge_attr, batch)``.
        datamodule: ``ActivityDataModule`` instance (``setup()`` is called internally if needed).
        loss: ``Loss`` instance
        max_recall_k: Largest rank cutoff ``K`` of the headline ``avg_recall@K``.
        recall_ks: Rank cutoffs of the logged ``recall@k``.
        num_epochs: Number of training epochs.
        verbose: Non-zero enables the Lightning progress bar.
        name: Experiment name used for logging and checkpoint filename.
        lr: Initial learning rate.
        reg: Optional regularisation; ``"l1"`` adds L1 weight penalty.
        full_info: Passed to ``extract_features`` to optionally include home/distance features.
        model_save_dir: Directory for checkpoint and CSV log files; None disables both.
        fast_dev_run: If True, runs 1 train batch and 1 val batch then exits; result DataFrame is empty.
        overfit_batches: Number of batches to overfit on; 0 disables (normal training).
        weight_decay: Weight decay parameter to AdamW, defaults to 1e-4.
        schedule_lr: add a ReduceLROnPlateau scheduler to the optimizer, defaults to False.
        pop_mode: "none"=do not inject ``pop_logits``; "offset"=inject in the loss function, "feature"=inject as
            features to the model.
        use_home_pe: if true, add home-anchored positional encodings to features
        compile_model: use PyTorch Dynamo compilation on the model
        wandb_params: parameters to configure WandB logging.
        extra_hyperparams: run-specific values to log alongside the standard ones, (architecture
            settings that otherwise exist inside the model name, e.g. GNN ``depth``).
        debug: Flag that enables `OverfitDebugCallback` statistics printing at the start and end of training, defaults to False.
        run_tag: Suffix distinguishing repeated runs of the same model (e.g. per training seed). It
            qualifies the checkpoint filename and logger run name only; ``name`` still identifies
            the model in the returned frame, so repeated runs group together.

    Returns:
        Long-format DataFrame tagged by ``stage``: one ``stage="fit"`` row per epoch carrying
        ``train_loss`` and the ``val_*`` metrics, plus (when a test run executed) one
        ``stage="test"`` row carrying the ``test_*`` metrics and one ``stage="test_user"`` row per
        test user carrying that user's ``r_precision``. Columns: ``name``, ``stage``,
        ``epoch``, ``train_loss``, ``val_*``, ``test_*``, ``user_id``, ``n_pos``, ``r_precision``;
        cells absent for a row's stage are null. Empty when ``fast_dev_run`` is True.
    """
    name = name or model.__class__.__name__
    run_name = name if run_tag is None else f"{name}-{run_tag}"

    datamodule.setup()

    pop_logit = datamodule.pop_logit if pop_mode != "none" else None

    lit_model = ActivityGraphModule(
        model=model,
        lr=lr,
        loss=loss,
        reg=reg,
        full_info=full_info,
        use_demographics=use_demographics,
        max_recall_k=max_recall_k,
        recall_ks=recall_ks,
        weight_decay=weight_decay,
        schedule_lr=schedule_lr,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        pop_logit=pop_logit,
        pop_mode=pop_mode,
        use_home_pe=use_home_pe,
        compile_model=compile_model,
        store_score_vectors=store_score_vectors,
        log_train_ranking=log_train_ranking,
    )

    # Build the callbacks

    collector = EpochMetricsCollector()
    callbacks: list[L.Callback] = [collector, LearningRateMonitor(logging_interval="epoch")]

    if model_save_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(model_save_dir),
                filename=run_name,
                save_last=False,
                save_top_k=1,
                monitor=loss.monitor,
                mode=loss.monitor_mode,
            )
        )

    if debug:
        callbacks.append(OverfitDebugCallback())

    log_dir = str(model_save_dir) if model_save_dir is not None else "."

    # Configure logging

    if wandb_params and wandb_params.use_wandb:
        logger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=run_name,
            group=wandb_params.group,
            save_dir=log_dir,
            tags=wandb_params.tags(),
        )
        logger.log_hyperparams({
            "model": name,
            "run_tag": run_tag,
            "dataset": wandb_params.dataset_name,
            "experiment": wandb_params.experiment,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "reg": reg,
            "full_info": full_info,
            "use_demographics": use_demographics,
            "schedule_lr": schedule_lr,
            "overfit_batches": overfit_batches,
            "pop_mode": pop_mode,
            "use_home_pe": use_home_pe,
            **(extra_hyperparams or {}),
        })
    else:
        logger = CSVLogger(save_dir=log_dir, name=run_name)

    # Create trainer and fit

    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        gradient_clip_val=1.0,
        enable_progress_bar=bool(verbose),
        enable_model_summary=False,
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    # Process results

    try:
        if fast_dev_run:
            return pl.DataFrame()

        if model_save_dir is not None:
            trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

        aggregate_results = aggregate_frame(collector.rows, name)
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
        if wandb_params and wandb_params.use_wandb:
            import wandb

            wandb.finish()
