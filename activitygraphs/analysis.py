"""Analysis of experiment outputs: seed spreads, paired model comparisons, hop bands, home-zone density
  - Distribution of result metrics between seeds/runs;
  - Comparison of models paired up by users;
  - Metrics per hop-band
  - Density of home zones

Reads the parquet files written by ``experiments.py:save_results`` and reports the points above. All analyses use the
parquet files generated to ``reports/data`` only, except the home-zone analysis, which reads the built ``ActivityDataset``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy.stats import wilcoxon

from activitygraphs.config import AnalysisConfig
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.training import PER_USER_STAGE

DEFAULT_ANALYSIS = AnalysisConfig(
    reference_model="ConditionalNodeMarginal",
    main_metric="test_r_precision",
    per_user_metric="r_precision",
    occupancy_thresholds=[10, 25, 60],
    hop_models=None,
    run=None,
)


# =========================================
# Loading
# =========================================


def load_run(report_data_path: str | Path, name: str, run: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the aggregate and per-user result frames of a run.

    Args:
        report_data_path: path to the Parquet results tables (e.g. ``reports/data/``).
        name: Dataset name used in filename, e.g. ``"GenevaTPG"``.
        run: Run number ``n`` of ``{name}-results-{n}.parquet``. Uses latest available if None.

    Returns:
        Tuple ``(aggregate_results, per_user_results)``.
    """
    path = Path(report_data_path)
    run = _latest_run(path, name) if run is None else run

    aggregate_results = pl.read_parquet(path / f"{name}-results-{run}.parquet")
    per_user_result = pl.read_parquet(path / f"{name}-per-user-{run}.parquet")

    return aggregate_results, per_user_result


def _latest_run(path: Path, name: str) -> int:
    """Highest run number among ``{name}-results-{n}.parquet`` files in ``path``."""
    run_numbers = [
        int(run_num)
        for f in path.iterdir()
        if f.suffix == ".parquet" and f.name.startswith(f"{name}-results-")
        for run_num in [f.stem.split("-")[-1]]
        if run_num.isdecimal()
    ]

    if not run_numbers:
        raise FileNotFoundError(f"no {name}-results-*.parquet in {path}")

    return max(run_numbers)


# =========================================
# Aggregate metrics and paired per-user comparison
# =========================================


def aggregate_metrics(aggregate_results: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Mean +- SD of a test metric over all training seeds, one row per model.

    Baselines are deterministic (no seed), so function reports ``n_seeds = 1`` and null SD.

    Returns:
        Frame of ``name, n_seeds, mean, sd, min, max``, sorted by mean (descending).
    """
    test_rows = aggregate_results.filter(pl.col("stage") == "test")

    if metric not in test_rows.columns:
        raise KeyError(f"{metric} not in the results DF; found {sorted(test_rows.columns)}")

    return (
        test_rows
        .group_by("name")
        .agg(
            n_seeds=pl.col(metric).count(),
            mean=pl.col(metric).mean(),
            sd=pl.col(metric).std(),
            min=pl.col(metric).min(),
            max=pl.col(metric).max(),
        )
        .sort("mean", descending=True)
    )


def paired_comparison(
    per_user_results: pl.DataFrame,
    reference_model: str,
    metric: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """Comparison of every model against ``reference``, paired up by user.

    Pairs by ``user_id`` after averaging over seeds. For each model, reports the paired mean difference between its score and
    the reference model. Also reports confidence interval of difference (bootstrap over users) and a
    Wilcoxon signed-rank p-value (test to compare mean of two paired samples).

    Returns:
        Frame of ``name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``, sorted by ``mean_diff`` (desc).
    """
    scores = _average_metric_over_seeds(per_user_results, metric)
    model_names = [n for n in scores["name"].unique().sort() if n != reference_model]

    reference_scores = scores.filter(pl.col("name") == reference_model).select("user_id", ref_value=metric)
    if reference_scores.is_empty():
        raise KeyError(f"Reference model {reference_model!r} not found in the per-user frame")

    rng = np.random.default_rng(seed)
    rows = []

    for model_name in model_names:
        paired_scores = (
            scores
            .filter(pl.col("name") == model_name)
            .select("user_id", value=metric)
            .join(reference_scores, on="user_id")
        )

        difference = paired_scores["value"].to_numpy() - paired_scores["ref_value"].to_numpy()
        ci_lo, ci_hi = _bootstrap_ci(difference, rng, n_bootstrap)
        wilcoxon_p = float(wilcoxon(difference).pvalue) if np.any(difference != 0.0) else 1.0
        sd_per_seed = _diff_sd_per_seed(per_user_results, model_name, reference_model, metric)

        rows.append({
            "name": model_name,
            "n_users": len(difference),
            "mean_diff": float(difference.mean()),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "wilcoxon_p": wilcoxon_p,
            "per_seed_sd": sd_per_seed,
        })

    return pl.DataFrame(rows).sort("mean_diff", descending=True)


def home_coverage_summary(per_user_results: pl.DataFrame) -> pl.DataFrame:
    """Computes the proportion of test users with a home node that does not appear in the training set, one row for
    ``home_seen_in_train=0`` and one row for ``home_seen_in_train=1``."""
    users = per_user_results.filter(pl.col("stage") == PER_USER_STAGE).select("user_id", "home_seen_in_train").unique()

    return (
        users
        .group_by("home_seen_in_train")
        .agg(n_users=pl.len())
        .with_columns(share=pl.col("n_users") / pl.col("n_users").sum())
        .sort("home_seen_in_train", descending=True)
    )


def paired_comparison_by_home_coverage(
    per_user_results: pl.DataFrame,
    reference_model: str,
    metric: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """Two separate runs of ``paired_comparison``, one on the test users with a home node seen in training and the other
     on the test users with an unseen home.

    Returns:
        Frame of ``home_seen_in_train, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``.
    """
    if "home_seen_in_train" not in per_user_results.columns:
        raise KeyError("per-user frame has no home_seen_in_train column; re-run to regenerate it")

    results_by_home_seen = []
    for seen in (True, False):
        subset = per_user_results.filter(pl.col("home_seen_in_train") == seen)
        if subset.filter(pl.col("stage") == PER_USER_STAGE).is_empty():
            continue

        comparison = paired_comparison(subset, reference_model, metric, n_bootstrap, seed)
        results_by_home_seen.append(comparison.with_columns(home_seen_in_train=pl.lit(seen)))

    results: pl.DataFrame = pl.concat(results_by_home_seen)

    return results.select("home_seen_in_train", pl.all().exclude("home_seen_in_train")).sort(
        "home_seen_in_train", "mean_diff", descending=[True, True]
    )


def _average_metric_over_seeds(per_user: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Average the per-user metrics over that model's training seeds. Avoids inflating apparent sample size
    of users (duplicate across seeds).

    Returns:
        Frame of ``name, user_id, <metric>`` with one row per model and user.
    """
    return (
        per_user
        .filter(pl.col("stage") == PER_USER_STAGE)
        .group_by("name", "user_id")
        .agg(pl.col(metric).mean())
        .sort("name", "user_id")
    )


def _bootstrap_ci(differences: np.ndarray, rng: np.random.Generator, n_bootstrap: int, alpha: float = 0.05):
    """Percentile bootstrap interval for the mean of ``diff``, resampling users."""
    if len(differences) == 0:
        return float("nan"), float("nan")

    draws = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    means = differences[draws].mean(axis=1)

    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _diff_sd_per_seed(per_user_results: pl.DataFrame, model_name: str, reference_model: str, metric: str) -> float:
    """SD across seeds of the per-seed mean paired difference of ``model_name`` against ``reference_model``.

    Differences are paired by user within a seed. Users are matched across seeds by user_id. If reference model has
    several seeds, the seeds are matched between the reference and the comparison model.

    Returns:
        SD over seeds, or NaN when fewer than two seeds are shared.
    """
    rows = per_user_results.filter(pl.col("stage") == PER_USER_STAGE)
    model_scores = rows.filter(pl.col("name") == model_name).select("seed", "user_id", value=metric)
    reference_scores = rows.filter(pl.col("name") == reference_model).select("seed", "user_id", ref_value=metric)

    model_seeds = set(model_scores["seed"].drop_nulls())
    shared_seeds = model_seeds & set(reference_scores["seed"].drop_nulls())

    if shared_seeds:
        paired_scores = model_scores.join(reference_scores, on=["seed", "user_id"])
    else:
        reference_means = reference_scores.group_by("user_id").agg(pl.col("ref_value").mean())
        paired_scores = model_scores.join(reference_means, on="user_id")

    sd_per_seed = (
        paired_scores.group_by("seed").agg(mean_diff=(pl.col("value") - pl.col("ref_value")).mean())["mean_diff"].std()
    )

    return float(cast(float, sd_per_seed)) if sd_per_seed is not None else float("nan")


# =========================================
# Hop band analysis
# =========================================


def compute_hop_band_table(aggregate_results: pl.DataFrame, models: Sequence[str] | None = None) -> pl.DataFrame:
    """Hop-band metrics, mean +- sd over training seeds. Computes the results grouped by bands of number hops away
    from user's home. ``share_of_pos`` specifies number of positives in hop-band (i.e. how much of test metric result
    is in this band)

    Args:
        aggregate_results: Aggregate result frame from ``load_run``.
        models: Restrict to these model names, e.g. an MLP and a GNN. Defaults to all.

    Returns:
        Frame of ``name, band, metric, mean, sd, n_pos, share_of_pos``, ordered by band then model.
    """
    test_rows = aggregate_results.filter(pl.col("stage") == "test")
    if models is not None:
        test_rows = test_rows.filter(pl.col("name").is_in(list(models)))

    parsed_hop_columns = [_extract_band_and_metric(col) for col in test_rows.columns if col.startswith("test_hop_")]

    col_to_band = {col: band for col, band, _, _ in parsed_hop_columns}
    col_to_metric = {col: metric for col, _, _, metric in parsed_hop_columns}

    hop_cols = [col for col, _, _, _ in parsed_hop_columns]
    band_to_ordinal = {band: lower_bound for _, band, lower_bound, _ in parsed_hop_columns}

    avg_scores_over_seeds = (
        test_rows
        .select("name", "seed", *hop_cols)
        .unpivot(index=["name", "seed"], variable_name="column", value_name="value")
        .with_columns(
            band=pl.col("column").replace_strict(col_to_band),
            metric=pl.col("column").replace_strict(col_to_metric),
        )
        .group_by("name", "band", "metric")
        .agg(mean=pl.col("value").mean(), sd=pl.col("value").std())
    )

    n_pos = avg_scores_over_seeds.filter(pl.col("metric") == "n_pos").select("name", "band", n_pos="mean")
    totals = n_pos.group_by("name").agg(total=pl.col("n_pos").sum())

    return (
        avg_scores_over_seeds
        .filter(pl.col("metric") != "n_pos")
        .join(n_pos, on=["name", "band"])
        .join(totals, on="name")
        .with_columns(share_of_pos=pl.col("n_pos") / pl.col("total"))
        .drop("total")
        .sort(pl.col("band").replace_strict(band_to_ordinal), "metric", "name")
    )


def _extract_band_and_metric(col_name: str) -> tuple[str, str, int, str]:
    """Split ``test_hop_{band}_{metric}`` into ``(column, band, band lower bound, metric)``,
    e.g. ``("test_hop_3-5_recall@2", "3-5", 3, "recall@2")``."""
    band, _, metric = col_name.removeprefix("test_hop_").partition("_")
    lower_band_bound = int(band.partition("-")[0].partition("+")[0])
    return col_name, band, lower_band_bound, metric


# =========================================
# User zone distribution
# =========================================


def home_zone_summary(dataset: ActivityDataset, occupancy_thresholds: Sequence[int] | None = None) -> pl.DataFrame:
    """Density summary of a users-per-home-zone table (single row).
        - ``users_per_occupied_zone`` is the mean number of users per _occupied_ zone.
        - ``users_per_node`` is the mean number of users per zone.
        - ``p50/p90`` are the quantiles of the number of users per zone.
        - ``share_users_ge_{t}`` reports the proportion of all users that live in zones of at least ``t`` inhabitants.

    Returns:
        One-row frame of ``n_users, n_nodes, n_occupied_zones, users_per_occupied_zone,
        users_per_node, p50, p90, max, share_users_ge_10/25/60``.
    """

    n_nodes = dataset.num_nodes
    home_nodes = dataset.spatial_features[:, :, dataset.is_home_spatial_idx].argmax(dim=1)
    occupancy_thresholds = (
        DEFAULT_ANALYSIS.occupancy_thresholds if occupancy_thresholds is None else occupancy_thresholds
    )

    home_nodes_df = pl.DataFrame({"home_node": home_nodes.cpu().numpy()})
    users_per_home_zone = home_nodes_df.group_by("home_node").agg(n_users=pl.len())

    total_users = int(users_per_home_zone["n_users"].sum())
    num_occupied_zones = len(users_per_home_zone)

    user_share_in_threshold_exprs = {
        f"share_users_ge_{t}": pl.col("n_users").filter(pl.col("n_users") >= t).sum() / total_users
        for t in occupancy_thresholds
    }

    return users_per_home_zone.select(
        n_users=total_users,
        n_nodes=n_nodes,
        n_occupied_zones=num_occupied_zones,
        users_per_occupied_zone=total_users / num_occupied_zones,
        users_per_node=total_users / n_nodes,
        p50=pl.col("n_users").median(),
        p90=pl.col("n_users").quantile(0.9),
        max=pl.col("n_users").max().cast(pl.Int64),
        **user_share_in_threshold_exprs,
    )


# =========================================
# Report
# =========================================


def print_report(
    report_data_path: str | Path,
    name: str,
    run: int | None = None,
    analysis: AnalysisConfig | None = None,
) -> None:
    """Print results analyses for a run of the whole comparison experiment: aggregate results, paired comparison,
    and hop bands.
    """
    run = run if run is not None else _latest_run(Path(report_data_path), name)
    analysis = DEFAULT_ANALYSIS if analysis is None else analysis
    aggregate, per_user = load_run(report_data_path, name, run)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(f"\n=== {name} run {run} ===")

        print(f"\n-- {analysis.main_metric}, avg over seeds --")
        print(aggregate_metrics(aggregate, analysis.main_metric))

        print(f"\n-- paired per-user {analysis.per_user_metric}, vs {analysis.reference_model} (avg over seeds) --")
        print(paired_comparison(per_user, analysis.reference_model, analysis.per_user_metric))

        print("\n-- hop bands (mean +- sd over seeds) --")
        print(compute_hop_band_table(aggregate, analysis.hop_models))

        if "home_seen_in_train" in per_user.columns:
            print("\n-- home coverage of test users --")
            print(home_coverage_summary(per_user))

            print(f"\n-- paired per-user {analysis.per_user_metric}, vs {analysis.reference_model}, by home coverage -")
            print(paired_comparison_by_home_coverage(per_user, analysis.reference_model, analysis.per_user_metric))
