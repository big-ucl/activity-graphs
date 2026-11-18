"""Upload of the distance-band reads of a recorded run to W&B, as one table for the custom charts of the workspace."""

import re
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

import wandb
from activitygraphs.runs import CONFIG_FILE

BAND_TABLE_KEY = "distance_bands"
"""Key of the band table in the W&B run, the table selected by the custom charts."""

RUN_DIR_PATTERN = re.compile(r"(?P<name>.+)-(?P<run>\d+)-(?P<started>\d{8}-\d{6})")


def band_seed_means(band_recall: pl.DataFrame, band_auc: pl.DataFrame | None) -> pl.DataFrame:
    """Return the mean recall and the mean within-band AUC of every model, seed and distance band.

    Args:
        band_recall: Per-user band recall from ``distance_band_recall``.
        band_auc: Per-user within-band AUC from ``within_band_auc``. None leaves the ``auc`` column null.

    Returns:
        Frame of ``model, seed, band, band_lo, band_recall, auc``, one row per model, seed and band, ordered by model,
        seed and band. ``band_recall`` is the mean over the users with a positive in the band, and ``auc`` the mean
        over the users scored in the band.
    """
    keys = ["name", "seed", "band", "band_lo"]
    recall_means = band_recall.group_by(keys).agg(band_recall=pl.col("band_recall").mean())

    if band_auc is None:
        seed_means = recall_means.with_columns(auc=pl.lit(None, dtype=pl.Float64))
    else:
        auc_means = band_auc.group_by(keys).agg(auc=pl.col("auc").mean())
        seed_means = recall_means.join(auc_means, on=keys, how="full", coalesce=True, nulls_equal=True)

    return (
        seed_means
        .rename({"name": "model"})
        .select("model", "seed", "band", "band_lo", "band_recall", "auc")
        .sort("model", "seed", "band_lo", nulls_last=True)
    )


def log_band_table(run_path: Path, band_means: pl.DataFrame) -> None:
    """Log ``band_means`` to a W&B run in the group of the training runs of the run at ``run_path``.

    The W&B run keeps one id per run directory, so a second call replaces the table of the first.

    Args:
        run_path: Directory of the recorded run, ``{name}-{N}-{YYYYmmdd-HHMMSS}``.
        band_means: Band means from ``band_seed_means``.
    """
    config = OmegaConf.load(run_path / CONFIG_FILE)
    group = _wandb_group(run_path.name, config.experiment.name)
    tags = [config.data.name, config.experiment.name, "analysis"]
    band_table = wandb.Table(dataframe=band_means.to_pandas())

    run = wandb.init(
        project=config.train.wandb_project,
        entity=config.train.wandb_entity,
        group=group,
        name=f"{run_path.name}-bands",
        id=f"{run_path.name}-bands",
        resume="allow",
        job_type="analysis",
        tags=tags,
    )
    run.log({BAND_TABLE_KEY: band_table})
    run.finish()


def _wandb_group(run_dir_name: str, experiment: str) -> str:
    """Return the W&B group of the training runs of the run directory ``run_dir_name``, as set by ``setup_experiment``.

    Raises:
        ValueError: ``run_dir_name`` is not ``{name}-{N}-{YYYYmmdd-HHMMSS}``.
    """
    match = RUN_DIR_PATTERN.fullmatch(run_dir_name)
    if match is None:
        raise ValueError(f"`{run_dir_name}` is not a numbered run directory, `{{name}}-{{N}}-{{YYYYmmdd-HHMMSS}}`.")

    return f"{match['name']}-{experiment}-{match['started']}"
