"""Result saving: split the long-format result frames of one experiment into its numbered parquet files."""

from pathlib import Path

import polars as pl

from activitygraphs.base import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE


def save_results(path: str | Path, name: str, *results: pl.DataFrame):
    """Concatenate result DataFrames and write them to ``<path>/data/``.

    The aggregate rows (``stage`` of ``fit``/``test``) go to ``<name>-results-<n>.parquet``, the
    per-user summary rows to ``<name>-per-user-<n>.parquet``, the full per-user score vectors to
    ``<name>-scores-<n>.parquet`` and the per-user distances from home to ``<name>-distances-<n>.parquet``.
    All four are numbered ``n``. They are split because they are
    different sizes, so only they only need to be loaded when required for analysis.
    """
    path = Path(path) / "data"

    ends = (f.stem.split("-")[-1] for f in path.iterdir() if f.suffix == ".parquet" and f.name.startswith(name))
    nums = (int(end) if end.isdecimal() else 0 for end in ends)
    max_num = max([0, *nums])

    combined = pl.concat(results, how="diagonal")
    per_user = combined.filter(pl.col("stage") == PER_USER_STAGE)
    score_vectors = combined.filter(pl.col("stage") == SCORE_VECTOR_STAGE)
    home_distances = combined.filter(pl.col("stage") == HOME_DISTANCE_STAGE)
    aggregate = combined.filter(~pl.col("stage").is_in([PER_USER_STAGE, SCORE_VECTOR_STAGE, HOME_DISTANCE_STAGE]))

    drop_empty_columns(aggregate).write_parquet(path / f"{name}-results-{max_num + 1}.parquet")

    if not per_user.is_empty():
        drop_empty_columns(per_user).write_parquet(path / f"{name}-per-user-{max_num + 1}.parquet")

    if not score_vectors.is_empty():
        drop_empty_columns(score_vectors).write_parquet(path / f"{name}-scores-{max_num + 1}.parquet")

    if not home_distances.is_empty():
        drop_empty_columns(home_distances).write_parquet(path / f"{name}-distances-{max_num + 1}.parquet")


def drop_empty_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that are null for every row (an artefact of the diagonal concat)."""
    if df.is_empty():
        return df

    return df.select([col for col in df.columns if df[col].null_count() < df.height])
