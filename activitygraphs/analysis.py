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
from scipy.stats import rankdata, wilcoxon

from activitygraphs.config import AnalysisConfig
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.metrics import DEFAULT_HOP_BANDS
from activitygraphs.ml.training import PER_USER_STAGE, SCORE_VECTOR_STAGE


# Aggregate column containing the home-included R-precision, kept as a sanity check beside the headline.
DIAGNOSTIC_METRIC_COLUMN = "test_r_precision_home_incl"

# Aggregate columns containing the number of users (used for average of main metric).
SCORED_USERS_COLUMN = "test_n_scored_users"
DROPPED_USERS_COLUMN = "test_n_dropped_users"

# Per-user column containing |RG_i| before exclusion of the home node.
REALISED_SIZE_COLUMN = "n_pos_home_incl"

# Per-user list column containing the hop distance of each positive, and the per-positive flag behind each per-user metric.
POSITIVE_HOPS_COLUMN = "pos_hops"
POSITIVE_FLAG_COLUMNS = {"r_precision": "pos_in_top_r", "recall": "pos_in_top_k"}

# Per-user list columns containing, for each positive, the number of scored candidates that scored higher than the
# positive and the number that tied with it.
N_SCORED_HIGHER_COLUMN = "pos_n_scored_higher"
N_TIED_COLUMN = "pos_n_tied"

DEFAULT_ANALYSIS = AnalysisConfig(
    reference_model="ConditionalNodeMarginal",
    main_metric="test_r_precision",
    diagnostic_metric=DIAGNOSTIC_METRIC_COLUMN,
    per_user_metric="r_precision",
    min_realised_size=3,
    occupancy_thresholds=[10, 25, 60],
    recall_curve_ks=[1, 2, 5, 10, 20],
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


def load_score_vectors(report_data_path: str | Path, name: str, run: int | None = None) -> pl.DataFrame:
    """Load the full per-user score vectors of a run: ``name, seed, user_id, scores``."""
    path = Path(report_data_path)
    run = _latest_run(path, name) if run is None else run
    scores_path = path / f"{name}-scores-{run}.parquet"

    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found; re-run with train.save_score_vectors=true")

    return pl.read_parquet(scores_path).filter(pl.col("stage") == SCORE_VECTOR_STAGE)


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


def _restrict_to_realised_size(per_user_results: pl.DataFrame, min_size: int) -> pl.DataFrame:
    """Keep only the test users whose realised set (incl. home node) has at least ``min_size`` nodes. Rows that are not
    per-user rows (the aggregate and score-vector stages) are not affected.
    """
    is_aggregate_stage = pl.col("stage") != PER_USER_STAGE
    is_realised_size_large_enough = pl.col(REALISED_SIZE_COLUMN) >= min_size

    return per_user_results.filter(is_aggregate_stage | is_realised_size_large_enough)


def _num_scored_dropped_users(aggregate_results: pl.DataFrame) -> tuple[int, int]:
    """Number of test users that were scored and dropped."""
    return (
        int(aggregate_results[SCORED_USERS_COLUMN].drop_nulls().first()),
        int(aggregate_results[DROPPED_USERS_COLUMN].drop_nulls().first()),
    )


def realised_size_summary(per_user_results: pl.DataFrame, n_dropped: int) -> pl.DataFrame:
    """Summary of the distribution of realised set sizes |RG_i| by number of users.

    Returns:
        Frame of ``n_pos_home_incl, scored, n_users, share_of_users``, sorted by size.
    """
    user_realised_set_sizes = (
        per_user_results.filter(pl.col("stage") == PER_USER_STAGE).select("user_id", REALISED_SIZE_COLUMN).unique()
    )

    # Frame of num_users by |RG_i|, does not include the dropped users
    num_users_by_realised_size = (
        user_realised_set_sizes.group_by(REALISED_SIZE_COLUMN).agg(n_users=pl.len()).with_columns(scored=pl.lit(True))
    )

    # Add the dropped users back in (only dropped when only home node was visited, so realised size is 1)
    num_dropped_users = pl.DataFrame(
        {REALISED_SIZE_COLUMN: [1], "n_users": [n_dropped], "scored": [False]},
        schema_overrides={
            REALISED_SIZE_COLUMN: num_users_by_realised_size.schema[REALISED_SIZE_COLUMN],
            "n_users": pl.UInt32,
        },
    )

    combined = (
        pl.concat([num_users_by_realised_size, num_dropped_users]) if n_dropped > 0 else num_users_by_realised_size
    )

    return combined.with_columns(share_of_users=pl.col("n_users") / pl.col("n_users").sum()).sort(
        REALISED_SIZE_COLUMN, "scored"
    )


def per_user_metric_summary(per_user_results: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Mean +- sd over training seeds of a per-user metric, one row per model (per-user counterpart of
    ``aggregate_metrics``), for reading the headline on a subset of the test users (e.g. where``|RG_i| >= 3``).

    Returns:
        Frame of ``name, n_users, n_seeds, mean, sd``, sorted by mean (descending).
    """
    per_user_rows = per_user_results.filter(pl.col("stage") == PER_USER_STAGE)
    avg_metric_by_seed = per_user_rows.group_by("name", "seed").agg(n_users=pl.len(), seed_mean=pl.col(metric).mean())

    return (
        avg_metric_by_seed
        .group_by("name")
        .agg(
            n_users=pl.col("n_users").max(),
            n_seeds=pl.len(),
            mean=pl.col("seed_mean").mean(),
            sd=pl.col("seed_mean").std(),
        )
        .sort("mean", descending=True)
    )


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
# Model health checks
# =========================================


def check_overfit_health(aggregate_results: pl.DataFrame) -> pl.DataFrame:
    """Checks to see if the models are capable of overfitting. Returns the best and final RPrecision on
    the training set. If working, ``train_r_precision`` should reach ~1.0.

    Returns:
        Frame of ``name, epochs, best_train_r_precision, final_train_r_precision``, sorted from worst to best.
    """
    if "train_r_precision" not in aggregate_results.columns:
        raise KeyError("no train_r_precision column; ``overfit_health`` needs a run with log_train_ranking enabled")

    fit_rows = aggregate_results.filter((pl.col("stage") == "fit") & pl.col("train_r_precision").is_not_null())

    return (
        fit_rows
        .sort("epoch")
        .group_by("name")
        .agg(
            epochs=pl.len(),
            best_train_r_precision=pl.col("train_r_precision").max(),
            final_train_r_precision=pl.col("train_r_precision").last(),
        )
        .sort("best_train_r_precision")
    )


def check_user_invariance(scores: np.ndarray, max_users: int = 200, seed: int = 0) -> float:
    """Check if the model gives users different scores or not. Computes the pairwise Spearman correlation between test
    users' score vectors.

    If ~1.0, then the model ranks every user identically: collapsed to a single global ordering and is ignoring
    ``is_home``/distance/demographics. Computes at most ``max_users`` pairs, sampled from the test set.
    """
    if len(scores) < 2:
        return float("nan")

    if len(scores) > max_users:
        rng = np.random.default_rng(seed)
        scores = scores[rng.choice(len(scores), size=max_users, replace=False)]

    unit_ranks = _row_ranks(scores)
    correlations = unit_ranks @ unit_ranks.T

    return float(np.median(correlations[np.triu_indices(len(scores), k=1)]))


def check_popularity_correlation(scores: np.ndarray, popularity: np.ndarray) -> float:
    """Check if the models are just learning the global/per-home-zone popularity or if they are extracting signal
    beyond that. Computes the Spearman between the user-averaged score vector and the training set popularity ranking.

    If ~1.0, then the model outputs the popularity marginal. Can work for ``logit(p_n)`` or the scores from NodeMarginal
    since the two are monotonic transformations of the popularities.
    """
    if len(scores) == 0:
        return float("nan")

    unit_ranks = _row_ranks(np.stack([scores.mean(axis=0), popularity]))

    return float(unit_ranks[0] @ unit_ranks[1])


def check_home_node_rank(scores: np.ndarray, home_nodes: np.ndarray) -> tuple[float, float]:
    """Check if the model ranks each user's home node at the top.

    Returns:
        ``(share of users whose home node ranks first, median home-node rank)``.
    """
    if len(scores) == 0:
        return float("nan"), float("nan")

    home_scores = scores[np.arange(len(scores)), home_nodes][:, None]
    ranks = 1 + (scores > home_scores).sum(axis=1)

    return float((ranks == 1).mean()), float(np.median(ranks))


def check_model_health(
    score_vectors: pl.DataFrame,
    per_user_results: pl.DataFrame,
    popularity_model: str = "NodeMarginal",
    max_users: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    """Performs three health checks to make sure each model are learning appropriately: 1) User invariance;
    2) Popularity correlation; 3) Home node rank.

    Args:
        score_vectors: DataFrame of per-user score vectors from ``load_score_vectors``.
        per_user_results: Per-user results frame from ``load_run``.
        popularity_model: Model whose score vector stands in for the train popularity ordering.
        max_users: max number of user pairs for correlation tests.
        seed: seed for the ``max_users`` subsample.

    Returns:
        Frame of ``name, n_seeds, user_invariance, popularity_corr, home_is_top1,
        home_median_rank`` and the ``*_sd`` standard deviation columns, sorted by ``user_invariance``
        descending (so that collapsed model at top).
    """
    home_by_user = per_user_results.filter(pl.col("stage") == PER_USER_STAGE).select("user_id", "home_node").unique()

    model_seeds = score_vectors.select("name", "seed").unique().sort("name", "seed")
    popularity_rows = _score_matrix(score_vectors, popularity_model, None)

    if len(popularity_rows) == 0:
        raise KeyError(f"popularity reference {popularity_model} has no score vectors in this run")

    rows = []

    for name, model_seed in model_seeds.iter_rows():
        scores = _score_matrix(score_vectors, name, model_seed)
        users = score_vectors.filter(
            (pl.col("name") == name)
            & (pl.col("seed").is_null() if model_seed is None else pl.col("seed") == model_seed)
        ).select("user_id")

        home_nodes = users.join(home_by_user, on="user_id", how="left")["home_node"].to_numpy()

        user_invariance = check_user_invariance(scores, max_users, seed)
        popularity_corr = check_popularity_correlation(scores, popularity_rows[0])
        home_is_top1, home_median_rank = check_home_node_rank(scores, home_nodes)

        rows.append({
            "name": name,
            "seed": model_seed,
            "user_invariance": user_invariance,
            "popularity_corr": popularity_corr,
            "home_is_top1": home_is_top1,
            "home_median_rank": home_median_rank,
        })

    checks = ["user_invariance", "popularity_corr", "home_is_top1", "home_median_rank"]

    return (
        pl
        .DataFrame(rows)
        .group_by("name")
        .agg(
            pl.len().alias("n_seeds"),
            *[pl.col(check).mean().alias(check) for check in checks],
            *[pl.col(check).std().alias(f"{check}_sd") for check in checks],
        )
        .sort("user_invariance", "name", descending=[True, False])
    )


def _score_matrix(score_vectors: pl.DataFrame, name: str, seed: int | None) -> np.ndarray:
    """Extracts the ``[n_users, num_nodes]`` matrix of user scores output by one model ``name`` and seed ``seed`` pair.
    Sorted in same order as the ``score_vectors`` frame."""
    rows = score_vectors.filter(pl.col("name") == name)
    rows = rows.filter(pl.col("seed").is_null() if seed is None else pl.col("seed") == seed)

    return np.asarray(rows["scores"].to_list(), dtype=np.float64)


def _row_ranks(scores: np.ndarray) -> np.ndarray:
    """Computes the ranks of each row (corrected for ties) and z-scores them, so a dot product over nodes is a Spearman
    correlation."""
    row_ranks = rankdata(scores, axis=-1)
    centred_row_ranks = row_ranks - row_ranks.mean(axis=-1, keepdims=True)

    return centred_row_ranks / np.linalg.norm(centred_row_ranks, axis=-1, keepdims=True).clip(min=1e-12)


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
    return col_name, band, _extract_band_lower_bound(band), metric


def _extract_band_lower_bound(band: str) -> int:
    """Return the lower hop bound of a band label, e.g. ``3`` for ``"3-5"`` and ``13`` for ``"13+"``."""
    return int(band.partition("-")[0].partition("+")[0])


def _hop_band(hops: pl.Expr) -> pl.Expr:
    """Return the label of the ``DEFAULT_HOP_BANDS`` band each hop distance falls in."""
    return pl.coalesce([
        pl.when(hops.is_between(low, high)).then(pl.lit(label)) for label, low, high in DEFAULT_HOP_BANDS
    ])


def band_contributions(per_user_results: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Split each user's per-user ``metric`` into the contributions of the positives in each hop band.

    Returns:
        Frame of ``name, stage, seed, user_id, band, contribution, n_band_pos``, one row per model, seed, user and band.
        ``contribution`` is the number of the band's positives ranked inside the metric's cutoff divided by the user's
        ``n_pos``, zero where the user has no positive in the band.

    Raises:
        KeyError: If ``metric`` has no per-positive flag column.
    """
    if metric not in POSITIVE_FLAG_COLUMNS:
        raise KeyError(f"no per-positive flag for {metric!r}; decomposable metrics are {sorted(POSITIVE_FLAG_COLUMNS)}")

    flag = POSITIVE_FLAG_COLUMNS[metric]
    rows = per_user_results.filter(pl.col("stage") == PER_USER_STAGE).with_row_index("row")

    per_band = (
        rows
        .select("row", POSITIVE_HOPS_COLUMN, flag)
        .explode(POSITIVE_HOPS_COLUMN, flag)
        .with_columns(band=_hop_band(pl.col(POSITIVE_HOPS_COLUMN)))
        .group_by("row", "band")
        .agg(hits=pl.col(flag).sum(), n_band_pos=pl.len())
    )

    return (
        rows
        .select("row", "name", "stage", "seed", "user_id", "n_pos")
        .join(per_band.select("band").unique(), how="cross")
        .join(per_band, on=["row", "band"], how="left")
        .with_columns(
            contribution=pl.col("hits").fill_null(0) / pl.col("n_pos"),
            n_band_pos=pl.col("n_band_pos").fill_null(0),
        )
        .select("name", "stage", "seed", "user_id", "band", "contribution", "n_band_pos")
    )


def band_decomposition(per_user_results: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Per-user ``metric`` split by the hop band of each positive, mean +- sd over training seeds.

    Returns:
        Frame of ``name, band, n_seeds, mean, sd, n_pos, share_of_metric, share_of_pos``, ordered by band then mean
        (descending). ``share_of_metric`` is the band's fraction of the model's summed band means, ``share_of_pos`` its
        fraction of the positives.
    """
    seed_means = (
        band_contributions(per_user_results, metric)
        .group_by("name", "seed", "band")
        .agg(seed_mean=pl.col("contribution").mean(), n_pos=pl.col("n_band_pos").sum())
    )
    band_to_ordinal = {band: _extract_band_lower_bound(band) for band in seed_means["band"].unique()}

    return (
        seed_means
        .group_by("name", "band")
        .agg(
            n_seeds=pl.len(),
            mean=pl.col("seed_mean").mean(),
            sd=pl.col("seed_mean").std(),
            n_pos=pl.col("n_pos").mean(),
        )
        .with_columns(
            share_of_metric=pl.col("mean") / pl.col("mean").sum().over("name"),
            share_of_pos=pl.col("n_pos") / pl.col("n_pos").sum().over("name"),
        )
        .sort(pl.col("band").replace_strict(band_to_ordinal), "mean", descending=[False, True])
    )


def paired_comparison_by_band(
    per_user_results: pl.DataFrame,
    reference_model: str,
    metric: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """``paired_comparison`` run on each hop band's contribution to the per-user ``metric``.

    Returns:
        Frame of ``band, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``, ordered by band then
        ``mean_diff`` (descending).
    """
    contributions = band_contributions(per_user_results, metric)
    bands = sorted(contributions["band"].unique(), key=_extract_band_lower_bound)

    comparisons = [
        paired_comparison(
            contributions.filter(pl.col("band") == band), reference_model, "contribution", n_bootstrap, seed
        ).with_columns(band=pl.lit(band))
        for band in bands
    ]

    return pl.concat(comparisons).select("band", pl.all().exclude("band"))


# =========================================
# Recall@k curve
# =========================================


def recall_at_ks(per_user_results: pl.DataFrame, ks: Sequence[int]) -> pl.DataFrame:
    """Per-user recall@k for each value of k ``ks``.

    If a positive is tied to enter the top-k, it is resolved as follows: instead of adding 1 to the count of hits,
    add ``# remaining empty top-k "slots" / # entries in tie``, the expected value of the hit count.

    Returns:
        Frame of ``name, stage, seed, user_id, k, recall_at_k``, one row per model, seed, user and cutoff.
    """
    rows = per_user_results.filter(pl.col("stage") == PER_USER_STAGE).with_row_index("row")
    cutoffs = pl.DataFrame({"k": list(ks)}, schema={"k": pl.Int64})

    return (
        rows
        .select("row", "name", "stage", "seed", "user_id", N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
        .explode(N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
        .join(cutoffs, how="cross")
        .with_columns(
            p_in_top_k=((pl.col("k") - pl.col(N_SCORED_HIGHER_COLUMN)) / (pl.col(N_TIED_COLUMN) + 1)).clip(0.0, 1.0)
        )
        .group_by("row", "name", "stage", "seed", "user_id", "k")
        .agg(recall_at_k=pl.col("p_in_top_k").mean())
        .sort("row", "k")
        .select("name", "stage", "seed", "user_id", "k", "recall_at_k")
    )


def recall_curve(per_user_results: pl.DataFrame, ks: Sequence[int]) -> pl.DataFrame:
    """Per-user recall@k at each k cutoff in ``ks``, mean +- sd over training seeds.

    Returns:
        Frame of ``name, k, n_seeds, mean, sd``, ordered by k then mean (descending).
    """
    return (
        recall_at_ks(per_user_results, ks)
        .group_by("name", "seed", "k")
        .agg(seed_mean=pl.col("recall_at_k").mean())
        .group_by("name", "k")
        .agg(n_seeds=pl.len(), mean=pl.col("seed_mean").mean(), sd=pl.col("seed_mean").std())
        .sort("k", "mean", descending=[False, True])
    )


def paired_comparison_by_k(
    per_user_results: pl.DataFrame,
    reference_model: str,
    ks: Sequence[int],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """``paired_comparison`` of the per-user recall@k at each cutoff in ``ks``.

    Returns:
        Frame of ``k, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``, ordered by k then
        ``mean_diff`` (descending).
    """
    recalls = recall_at_ks(per_user_results, ks)

    comparisons = [
        paired_comparison(
            recalls.filter(pl.col("k") == k), reference_model, "recall_at_k", n_bootstrap, seed
        ).with_columns(k=pl.lit(k, dtype=pl.Int64))
        for k in ks
    ]

    return pl.concat(comparisons).select("k", pl.all().exclude("k"))


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
    report_data_path: Path = Path(report_data_path)
    run = run if run is not None else _latest_run(report_data_path, name)
    analysis = DEFAULT_ANALYSIS if analysis is None else analysis
    aggregate, per_user = load_run(report_data_path, name, run)

    n_scored, n_dropped = _num_scored_dropped_users(aggregate)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(f"\n=== {name} run {run} ===")
        print(f"test users: {n_scored} scored, {n_dropped} dropped due to empty RG_i")

        print(f"\n-- {analysis.main_metric}, avg over seeds --")
        print(aggregate_metrics(aggregate, analysis.main_metric))

        print(f"\n-- {analysis.diagnostic_metric}, avg over seeds (sanity check) --")
        print(aggregate_metrics(aggregate, analysis.diagnostic_metric))

        print(f"\n-- paired per-user {analysis.per_user_metric}, vs {analysis.reference_model} (avg over seeds) --")
        print(paired_comparison(per_user, analysis.reference_model, analysis.per_user_metric))

        min_size = analysis.min_realised_size
        restricted = _restrict_to_realised_size(per_user, min_size)

        print("\n-- test users by |RG_i| --")
        print(realised_size_summary(per_user, n_dropped))

        print(f"\n-- per-user {analysis.per_user_metric} where |RG_i| >= {min_size}, avg over seeds --")
        print(per_user_metric_summary(restricted, analysis.per_user_metric))

        print(
            f"\n-- paired per-user {analysis.per_user_metric} where |RG_i| >= {min_size}, vs {analysis.reference_model} --"
        )
        print(paired_comparison(restricted, analysis.reference_model, analysis.per_user_metric))

        print("\n-- hop bands (mean +- sd over seeds) --")
        print(compute_hop_band_table(aggregate, analysis.hop_models))

        print(f"\n-- per-user {analysis.per_user_metric} by hop band of each positive, avg over seeds --")
        print(band_decomposition(per_user, analysis.per_user_metric))

        print(
            f"\n-- paired per-user {analysis.per_user_metric} by hop band of each positive, vs {analysis.reference_model} --"
        )
        print(paired_comparison_by_band(per_user, analysis.reference_model, analysis.per_user_metric))

        ks = analysis.recall_curve_ks
        print(f"\n-- per-user recall@k at k = {ks}, avg over seeds --")
        print(recall_curve(per_user, ks))

        print(f"\n-- paired per-user recall@k at k = {ks}, vs {analysis.reference_model} --")
        print(paired_comparison_by_k(per_user, analysis.reference_model, ks))

        print("\n-- home coverage of test users --")
        print(home_coverage_summary(per_user))

        print(f"\n-- paired per-user {analysis.per_user_metric}, vs {analysis.reference_model}, by home coverage --")
        print(paired_comparison_by_home_coverage(per_user, analysis.reference_model, analysis.per_user_metric))

        if (report_data_path / f"{name}-scores-{run}.parquet").exists():
            print("\n-- model health checks --")
            print(check_model_health(load_score_vectors(report_data_path, name, run), per_user))
