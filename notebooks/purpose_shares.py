"""Share of each survey's non-home training labels that are work or study locations.

Run from the repo root: ``uv run python notebooks/purpose_shares.py [tpg] [cmap]``.
"""

import sys
from pathlib import Path

import polars as pl

from activitygraphs.base import Purpose
from activitygraphs.config import read_config
from activitygraphs.dataprocessing import load_survey_and_graph

ANCHOR_PURPOSES = [Purpose.WORK_MAIN.value, Purpose.STUDY.value]


def non_home_positives(location_visits: pl.DataFrame, home_locations: pl.DataFrame) -> pl.DataFrame:
    """Build one row per user and non-home visited location, with the purposes recorded there.

    Args:
        location_visits: Visits with columns ``user_id``, ``purpose`` and ``loc_id``.
        home_locations: One home per user, with columns ``user_id`` and ``loc_id``.

    Returns:
        DataFrame with columns ``user_id``, ``loc_id``, ``purposes`` (list of purpose strings),
        ``is_anchor_only`` (every purpose is work or study) and ``weight``, the positive's share of
        the user-averaged recall: ``1 / (n_pos_i * n_users)``, summing to 1.
    """
    homes = home_locations.select("user_id", home_loc_id="loc_id")
    positives = (
        location_visits
        .join(homes, on="user_id")
        .filter(pl.col("loc_id") != pl.col("home_loc_id"))
        .group_by("user_id", "loc_id")
        .agg(purposes=pl.col("purpose").cast(pl.String).unique().sort())
    )
    n_users = positives["user_id"].n_unique()

    return positives.with_columns(
        is_anchor_only=pl.col("purposes").list.eval(pl.element().is_in(ANCHOR_PURPOSES)).list.all(),
        weight=1 / (pl.len().over("user_id") * n_users),
    )


def purpose_table(positives: pl.DataFrame) -> pl.DataFrame:
    """Tabulate how often each purpose is recorded at a non-home positive.

    A positive visited for several purposes counts once under each, so the shares sum to more than 1.

    Returns:
        DataFrame with one row per purpose: ``n_positives``, ``share_of_positives`` and
        ``share_of_metric_weight``.
    """
    n_positives = positives.height

    return (
        positives
        .explode("purposes")
        .group_by(purpose="purposes")
        .agg(n_positives=pl.len(), share_of_metric_weight=pl.col("weight").sum())
        .with_columns(share_of_positives=pl.col("n_positives") / n_positives)
        .select("purpose", "n_positives", "share_of_positives", "share_of_metric_weight")
        .sort("n_positives", descending=True)
    )


def summary_table(positives: pl.DataFrame) -> pl.DataFrame:
    """Summarise how much of the label set and of the headline metric sits on work and study locations.

    Returns:
        Two-column DataFrame of ``quantity`` and ``value``.
    """
    has_work = pl.col("purposes").list.contains(Purpose.WORK_MAIN.value)
    has_study = pl.col("purposes").list.contains(Purpose.STUDY.value)
    has_anchor = has_work | has_study

    per_user = positives.group_by("user_id").agg(
        n_pos=pl.len(),
        any_work=has_work.any(),
        any_anchor=has_anchor.any(),
        all_anchor_only=pl.col("is_anchor_only").all(),
    )

    def weight_where(cond: pl.Expr) -> float:
        return positives.filter(cond)["weight"].sum()

    rows = {
        "users with >= 1 non-home positive": per_user.height,
        "non-home positives": positives.height,
        "mean non-home positives per user": per_user["n_pos"].mean(),
        "share of positives with a work purpose": positives.select(has_work.mean()).item(),
        "share of positives with a study purpose": positives.select(has_study.mean()).item(),
        "share of positives with work or study": positives.select(has_anchor.mean()).item(),
        "share of positives with only work/study": positives["is_anchor_only"].mean(),
        "metric weight on positives with work or study": weight_where(has_anchor),
        "metric weight on positives with only work/study": weight_where(pl.col("is_anchor_only")),
        "share of users with a work positive": per_user["any_work"].mean(),
        "share of users with a work or study positive": per_user["any_anchor"].mean(),
        "share of users whose positives are all work/study only": per_user["all_anchor_only"].mean(),
    }

    return pl.DataFrame({"quantity": list(rows), "value": [float(v) for v in rows.values()]})


def main(datasets: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_float_precision(4)

    for name in datasets:
        cfg = read_config(project_root, verbose=False, data=name)
        data, _, _ = load_survey_and_graph(cfg.data, project_root)
        positives = non_home_positives(data.location_visits, data.home_locations)

        print(f"\n=== {cfg.data.name}: all modelled users, home node excluded ===")
        print(summary_table(positives))
        print(purpose_table(positives))


if __name__ == "__main__":
    main(sys.argv[1:] or ["tpg", "cmap"])
