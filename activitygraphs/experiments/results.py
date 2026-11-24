"""This module creates the run directory, records the config used by the run and writes the result frames to parquet."""

from pathlib import Path
from typing import cast

import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from activitygraphs.base import FITTED_PARAM_STAGE, HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config, ModelConfig
from activitygraphs.runs import (
    CONFIG_FILE,
    DISTANCES_FILE,
    FITTED_PARAMS_FILE,
    HYDRA_LINK,
    PER_USER_FILE,
    RESULTS_FILE,
    SCORES_FILE,
    create_run_dir,
)

KEPT_EMPTY_COLUMNS = ("seed",)
"""Columns kept by ``save_results`` even when null for every row, since the analysis groups by them."""


def create_output_dir(cfg: Config, run: str) -> Path | None:
    """Create the directory of the run of id ``run``.

    Returns:
        A new directory under ``cfg.paths.runs``, or under ``cfg.paths.overfit`` when ``cfg.trainer.overfit_batches > 0``.
        None on a ``fast_dev_run``, which records nothing.
    """
    if cfg.trainer.fast_dev_run:
        return None
    if cfg.trainer.overfit_batches > 0:
        return create_run_dir(Path(cfg.paths.overfit), run)

    return create_run_dir(Path(cfg.paths.runs), run)


def run_config(cfg: Config) -> dict:
    """Resolve the run config without ``models``. The architecture of each variant is logged separately."""
    config = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    config.pop("models")

    return config


def save_config(run_dir: Path, cfg: Config, models: dict[str, ModelConfig]):
    """Write the config used by the run to ``run_dir/config.yaml``. Link the Hydra run directory beside the config.

    Args:
        run_dir: The run directory.
        cfg: The config used by the experiment.
        models: The architectures as trained, in place of ``cfg.models``.
    """
    record = OmegaConf.merge(cfg, {"models": models})
    OmegaConf.save(record, run_dir / CONFIG_FILE, resolve=True)

    if HydraConfig.initialized():
        (run_dir / HYDRA_LINK).symlink_to(Path(HydraConfig.get().runtime.output_dir).resolve())


def save_results(run_dir: Path, *results: pl.DataFrame):
    """Concatenate result DataFrames and write them to ``run_dir``.

    The aggregate rows (``stage`` of ``fit``/``test``) go to ``results.parquet``, the
    per-user summary rows to ``per-user.parquet``, the full per-user score vectors to
    ``scores.parquet``, the per-user distances from home to ``distances.parquet`` and the fitted parameters of the
    baselines to ``fitted-params.parquet``.
    They are split because they are
    different sizes, so only they only need to be loaded when required for analysis.
    """
    combined = pl.concat(results, how="diagonal")
    per_user = combined.filter(pl.col("stage") == PER_USER_STAGE)
    score_vectors = combined.filter(pl.col("stage") == SCORE_VECTOR_STAGE)
    home_distances = combined.filter(pl.col("stage") == HOME_DISTANCE_STAGE)
    fitted_params = combined.filter(pl.col("stage") == FITTED_PARAM_STAGE)
    split_stages = [PER_USER_STAGE, SCORE_VECTOR_STAGE, HOME_DISTANCE_STAGE, FITTED_PARAM_STAGE]
    aggregate = combined.filter(~pl.col("stage").is_in(split_stages))

    _drop_empty_columns(aggregate).write_parquet(run_dir / RESULTS_FILE)

    if not per_user.is_empty():
        _drop_empty_columns(per_user).write_parquet(run_dir / PER_USER_FILE)

    if not score_vectors.is_empty():
        _drop_empty_columns(score_vectors).write_parquet(run_dir / SCORES_FILE)

    if not home_distances.is_empty():
        _drop_empty_columns(home_distances).write_parquet(run_dir / DISTANCES_FILE)

    if not fitted_params.is_empty():
        _drop_empty_columns(fitted_params).write_parquet(run_dir / FITTED_PARAMS_FILE)


def _drop_empty_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that are null for every row (an artefact of the diagonal concat), except ``KEPT_EMPTY_COLUMNS``."""
    if df.is_empty():
        return df

    kept_cols = [col for col in df.columns if col in KEPT_EMPTY_COLUMNS or df[col].null_count() < df.height]

    return df.select(kept_cols)
