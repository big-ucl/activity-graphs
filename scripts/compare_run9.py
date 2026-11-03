"""One-off: compare a new CMAP core run per user against CMAP run 9 (run 9 still has the old ``user_id`` column).

Usage, from activity-graphs/:
    uv run python scripts/compare_run9.py reports/runs/CMAP-9-20261030-163713 reports/runs/CMAP-<N>-<timestamp>
"""

import sys
from pathlib import Path

import polars as pl

from activitygraphs.analysis import with_ranking_metrics
from activitygraphs.runs import PER_USER_FILE

MAX_RECALL_K = 50


def per_user(run: Path, rename: bool) -> pl.DataFrame:
    """Return the per-user ``avg_recall`` of every model of ``run``, renaming run 9's ``user_id`` when ``rename``."""
    frame = pl.read_parquet(run / PER_USER_FILE)
    if rename:
        frame = frame.rename({"user_id": "user_idx"})
    frame = with_ranking_metrics(frame, MAX_RECALL_K)
    return frame.select("name", "seed", "is_baseline", "user_idx", "avg_recall")


old = per_user(Path(sys.argv[1]), rename=True)
new = per_user(Path(sys.argv[2]), rename=False)

baselines = old.filter(pl.col("is_baseline")).join(
    new.filter(pl.col("is_baseline")),
    on=["name", "seed", "user_idx"],
    how="full",
    suffix="_new",
    coalesce=True,
    nulls_equal=True,
)
print("Baselines: largest per-user |avg_recall difference| (0 means identical)")
print(
    baselines
    .group_by("name")
    .agg(
        n_users=pl.len(),
        unmatched=(pl.col("avg_recall").is_null() | pl.col("avg_recall_new").is_null()).sum(),
        max_abs_diff=(pl.col("avg_recall_new") - pl.col("avg_recall")).abs().max(),
    )
    .sort("name")
)

learned_old = old.filter(~pl.col("is_baseline")).group_by("name", "user_idx").agg(pl.col("avg_recall").mean())
learned_new = new.filter(~pl.col("is_baseline")).group_by("name", "user_idx").agg(pl.col("avg_recall").mean())
learned = learned_old.join(learned_new, on=["name", "user_idx"], suffix="_new")
print("Learned variants: seed-mean avg_recall, paired per user (new - run 9); run-to-run spread is about 0.003")
print(
    learned
    .group_by("name")
    .agg(
        n_users=pl.len(),
        run9=pl.col("avg_recall").mean(),
        new=pl.col("avg_recall_new").mean(),
        mean_diff=(pl.col("avg_recall_new") - pl.col("avg_recall")).mean(),
    )
    .sort("name")
)
