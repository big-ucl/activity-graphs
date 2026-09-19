"""Analysis of experiment outputs: seed spreads, paired model comparisons, hop bands, distance bands, home-zone density
  - Distribution of result metrics between seeds/runs;
  - Comparison of models paired up by users;
  - Metrics per hop-band
  - Per-user avg_recall by distance band of each positive
  - AUC of each positive within its distance band from home
  - Density of home zones

Reads the parquet files written by ``experiments.py:save_results`` and reports the points above. All analyses use the
parquet files generated to ``reports/data`` only, except the home-zone analysis, which reads the built ``ActivityDataset``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata, wilcoxon

from activitygraphs.config import AnalysisConfig
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.metrics import DEFAULT_HOP_BANDS
from activitygraphs.ml.ranking import average_recall, expected_recall_at_k, positive_band_auc
from activitygraphs.ml.training import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE


# Baseline model representing node popularity.
POPULARITY_MODEL = "VisitFrequency"

# Aggregate column containing the R-precision, reported as a diagnostic beside the headline.
R_PRECISION_COLUMN = "test_r_precision"

# Aggregate columns containing the number of users (used for average of main metric).
SCORED_USERS_COLUMN = "test_n_scored_users"
DROPPED_USERS_COLUMN = "test_n_dropped_users"

# Per-user column containing |RG_i| before exclusion of the home node.
REALISED_SIZE_COLUMN = "n_pos_home_incl"

# Per-user column containing mean recall@k over k = 1..budget.
AVG_RECALL_COLUMN = "avg_recall"

# Per-user list column containing the hop distance of each positive
PER_POSITIVE_HOPS_COLUMN = "pos_hops"
# Per-user list column containing each positive's contribution to ``avg_recall``.
PER_POSITIVE_AVG_RECALL_COLUMN = "pos_avg_recall"
# Dict of per-user metric -> corresponding value for each positive
PER_POSITIVE_VALUE_COLUMNS = {"r_precision": "pos_in_top_r", AVG_RECALL_COLUMN: PER_POSITIVE_AVG_RECALL_COLUMN}

# Per-user list columns containing, for each positive, the number of scored candidates that scored higher than the
# positive and the number that tied with it.
N_SCORED_HIGHER_COLUMN = "pos_n_scored_higher"
N_TIED_COLUMN = "pos_n_tied"

# Per-user list column containing the network node index of each positive, and per-user column containing the user's
# home node. Together with the score vectors and the home distances they carry the within-band AUC.
PER_POSITIVE_NODE_COLUMN = "pos_node"
HOME_NODE_COLUMN = "home_node"

METRES_PER_KM = 1000.0

# Users per within-band AUC pass. The comparison holds a ``[n_positives, n_nodes]`` mask per chunk, so chunking keeps
# a full test split out of memory.
USER_CHUNK = 512

DEFAULT_ANALYSIS = AnalysisConfig(
    reference_model="Gravity",
    main_metric="test_avg_recall@50",
    per_user_metric=AVG_RECALL_COLUMN,
    min_realised_size=3,
    occupancy_thresholds=[10, 25, 60],
    recall_curve_ks=[1, 3, 5, 10, 20, 50],
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


def load_home_distances(report_data_path: str | Path, name: str, run: int | None = None) -> pl.DataFrame:
    """Load each test user's distance from home to every node of a run: ``user_id, distances``, in metres."""
    path = Path(report_data_path)
    run = _latest_run(path, name) if run is None else run

    distances = pl.read_parquet(path / f"{name}-distances-{run}.parquet")

    return distances.filter(pl.col("stage") == HOME_DISTANCE_STAGE)


def with_ranking_metrics(per_user_results: pl.DataFrame, budget: int) -> pl.DataFrame:
    """Add the per-user ``avg_recall`` and per-positive ``pos_avg_recall`` columns, derived from the stored rank counts.

    Args:
        per_user_results: Per-user results frame from ``load_run``.
        budget: Largest rank cutoff ``K`` of ``avg_recall@K``.

    Returns:
        ``per_user_results`` with ``avg_recall`` and ``pos_avg_recall`` columns.
    """
    positives = (
        per_user_results
        .select(N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
        .with_row_index("row")
        .explode(N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
    )
    per_user = (
        positives
        .select("row", contribution=_positive_average_recall(positives, budget))
        .group_by("row", maintain_order=True)
        .agg(pl.col("contribution"))
        .sort("row")
    )

    return per_user_results.with_columns(
        per_user["contribution"].list.mean().alias(AVG_RECALL_COLUMN),
        per_user["contribution"].alias(PER_POSITIVE_AVG_RECALL_COLUMN),
    )


def _positive_average_recall(positives: pl.DataFrame, budget: int) -> pl.Series:
    """Per-positive ``average_recall`` of a frame with one row per positive."""
    return pl.Series(
        average_recall(positives[N_SCORED_HIGHER_COLUMN].to_numpy(), positives[N_TIED_COLUMN].to_numpy(), budget),
        dtype=pl.Float64,
    )


def append_baselines(results: pl.DataFrame, extra_results: pl.DataFrame) -> pl.DataFrame:
    """Append the baseline rows of another run of the same dataset to a result frame.

    Their users pair with the other run's by ``user_id``, since the split is fixed by ``train.split_seed``.

    Args:
        results: Aggregate, per-user or score-vector frame of the main run.
        extra_results: The same kind of frame from the other run.

    Returns:
        ``results`` with the other run's baseline rows appended.

    Raises:
        ValueError: If a baseline name already appears in ``results``.
    """
    baselines = extra_results.filter(pl.col("is_baseline"))
    shared = set(results["name"].unique()) & set(baselines["name"].unique())

    if shared:
        raise ValueError(f"models {sorted(shared)} appear in both runs")

    return pl.concat([results, baselines], how="diagonal_relaxed")


def load_report_run(
    report_data_path: str | Path, name: str, run: int, max_recall_k: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load a run for reporting with the aggregate results and the per-user ``avg_recall``.

    Args:
        report_data_path: path to the Parquet results tables (e.g. ``reports/data/``).
        name: Dataset name used in filename, e.g. ``"GenevaTPG"``.
        run: Run number of the main run.
        max_recall_k: Largest rank cutoff ``K`` of ``avg_recall@K``.

    Returns:
        Tuple ``(aggregate_results, per_user_results)``.
    """
    aggregate, per_user = load_run(report_data_path, name, run)
    return aggregate, with_ranking_metrics(per_user, max_recall_k)


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

    The closed-form baselines are fitted once, so the function reports ``n_seeds = 1`` and a null SD for them.

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


def check_overfit_health(aggregate_results: pl.DataFrame, budget: int) -> pl.DataFrame:
    """Checks to see if the models are capable of overfitting. Returns the best and final average recall on
    the training set. If working, ``train_avg_recall@K`` should reach ~1.0.

    Args:
        aggregate_results: Aggregate result frame of an ``overfit_health`` run.
        budget: Largest rank cutoff ``K`` of ``avg_recall@K``.

    Returns:
        Frame of ``name, epochs, best_train_avg_recall, final_train_avg_recall``, sorted from worst to best.
    """
    column = f"train_avg_recall@{budget}"
    if column not in aggregate_results.columns:
        raise KeyError(f"no {column} column; ``overfit_health`` needs a run with log_train_ranking enabled")

    fit_rows = aggregate_results.filter((pl.col("stage") == "fit") & pl.col(column).is_not_null())

    return (
        fit_rows
        .sort("epoch")
        .group_by("name")
        .agg(
            epochs=pl.len(),
            best_train_avg_recall=pl.col(column).max(),
            final_train_avg_recall=pl.col(column).last(),
        )
        .sort("best_train_avg_recall")
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

    If ~1.0, then the model outputs the popularity marginal. Can work for ``logit(p_n)`` or the scores from
    VisitFrequency since the two are monotonic transformations of the popularities.
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
    popularity_model: str,
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

    Args:
        per_user_results: Per-user results frame from ``load_report_run``.
        metric: A per-user metric with a per-positive value in ``POSITIVE_VALUE_COLUMNS``.

    Returns:
        Frame of ``name, stage, seed, user_id, band, contribution, n_band_pos``, one row per model, seed, user and band.
        ``contribution`` is how much the metric evaluated on only this band contributes to the overall metric value.
    Raises:
        KeyError: If ``metric`` has no per-positive value.
    """
    if metric not in PER_POSITIVE_VALUE_COLUMNS:
        raise KeyError(
            f"no per-positive value for {metric!r}; decomposable metrics are {sorted(PER_POSITIVE_VALUE_COLUMNS)}"
        )

    rows = per_user_results.filter(pl.col("stage") == PER_USER_STAGE).with_row_index("row")
    per_pos_value_col = PER_POSITIVE_VALUE_COLUMNS[metric]

    per_band = (
        rows
        .select("row", PER_POSITIVE_HOPS_COLUMN, per_pos_value_col)
        .explode(PER_POSITIVE_HOPS_COLUMN, per_pos_value_col)
        .select("row", PER_POSITIVE_HOPS_COLUMN, value=pl.col(per_pos_value_col).cast(pl.Float64))
        .with_columns(band=_hop_band(pl.col(PER_POSITIVE_HOPS_COLUMN)))
        .group_by("row", "band")
        .agg(band_sum=pl.col("value").sum(), n_band_pos=pl.len())
    )

    return (
        rows
        .select("row", "name", "stage", "seed", "user_id", "n_pos")
        .join(per_band.select("band").unique(), how="cross")
        .join(per_band, on=["row", "band"], how="left")
        .with_columns(
            contribution=pl.col("band_sum").fill_null(0.0) / pl.col("n_pos"),
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
# Distance band analysis
# =========================================


def _edge_label(edge: pl.Expr) -> pl.Expr:
    """Return a band edge in km as a string, without the trailing zero of a whole number of km."""
    return pl.when(edge == edge.round(0)).then(edge.cast(pl.Int64).cast(pl.String)).otherwise(edge.cast(pl.String))


def _distance_band_label(lower: pl.Expr, upper: pl.Expr) -> pl.Expr:
    """Return the label of the distance band with these edges in km, e.g. ``"1-2.5"`` or ``"10+"``."""
    finite_upper = pl.when(upper.is_infinite()).then(lower).otherwise(upper)

    return (
        pl
        .when(upper.is_infinite())
        .then(_edge_label(lower) + pl.lit("+"))
        .otherwise(_edge_label(lower) + pl.lit("-") + _edge_label(finite_upper))
    )


def distance_band_index(distances: np.ndarray, bands: Sequence[float]) -> np.ndarray:
    """Return the band each distance falls in, as a position in ``bands``.

    Args:
        distances: Distances from home in metres, any shape.
        bands: Upper band edges in km, each band holding the distances in ``[lower_edge, upper_edge)``.

    Returns:
        Integer array shaped like ``distances``.
    """
    return np.digitize(distances / METRES_PER_KM, np.asarray(bands, dtype=np.float64))


def _band_edges(bands: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower and upper edge in km of every band, the last band's upper edge being ``inf``."""
    edges = np.asarray(bands, dtype=np.float64)

    return np.concatenate([[0.0], edges]), np.concatenate([edges, [np.inf]])


def _per_positive_band_auc(
    scores: np.ndarray, positives: Sequence[Sequence[int]], home_nodes: np.ndarray, band_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score every positive of one model and seed against the unvisited nodes of its own band.

    Args:
        scores: Score of every node for every user, ``[n_users, n_nodes]``.
        positives: Node indices of each user's positives, one sequence per user.
        home_nodes: Home node of each user, ``[n_users]``.
        band_index: Band of every node for every user, ``[n_users, n_nodes]``.

    Returns:
        ``(auc, n_negatives)``, each ``[n_positives]``, one entry per positive in user order.
    """
    scored = []

    for start in range(0, len(scores), USER_CHUNK):
        chunk = slice(start, start + USER_CHUNK)
        chunk_scores = scores[chunk]

        target = np.zeros(chunk_scores.shape, dtype=bool)
        for user, user_positives in enumerate(positives[chunk]):
            target[user, list(user_positives)] = True

        valid = np.ones(chunk_scores.shape, dtype=bool)
        valid[np.arange(len(chunk_scores)), home_nodes[chunk]] = False

        scored.append(
            positive_band_auc(
                torch.from_numpy(chunk_scores),
                torch.from_numpy(target),
                torch.from_numpy(band_index[chunk]),
                torch.from_numpy(valid),
            )
        )

    return (
        np.concatenate([auc.numpy() for auc, _ in scored]),
        np.concatenate([n_negatives.numpy() for _, n_negatives in scored]),
    )


def distance_band_recall(
    per_user_results: pl.DataFrame, home_distances: pl.DataFrame, bands: Sequence[float]
) -> pl.DataFrame:
    """Each user's per-user ``avg_recall`` split by the distance band from home of each positive.

    Args:
        per_user_results: Per-user results frame from ``load_report_run``.
        home_distances: Per-user distances from home from ``load_home_distances``.
        bands: Upper band edges in km.

    Returns:
        Frame of ``name, stage, seed, user_id, band, band_lo, band_recall, n_band_pos, n_pos``, one row per model,
        seed, user and band the user has a positive in. ``band_recall`` is the mean ``pos_avg_recall`` of the user's
        positives in the band, ranked against every node, so ``band_recall * n_band_pos / n_pos`` summed over bands
        is the user's ``avg_recall``.
    """
    lower_edges, upper_edges = _band_edges(bands)
    distances = home_distances.select("user_id", "distances")

    positives = (
        per_user_results
        .filter(pl.col("stage") == PER_USER_STAGE)
        .join(distances, on="user_id")
        .with_columns(
            pos_distance=pl.col("distances").list.gather(pl.col(PER_POSITIVE_NODE_COLUMN)),
        )
        .select("name", "seed", "user_id", "n_pos", PER_POSITIVE_AVG_RECALL_COLUMN, "pos_distance")
        .explode(PER_POSITIVE_AVG_RECALL_COLUMN, "pos_distance")
    )
    band_index = distance_band_index(positives["pos_distance"].to_numpy().astype(np.float64), bands)

    return (
        positives
        .with_columns(band_lo=pl.Series(lower_edges[band_index]), band_hi=pl.Series(upper_edges[band_index]))
        .with_columns(band=_distance_band_label(pl.col("band_lo"), pl.col("band_hi")))
        .group_by("name", "seed", "user_id", "band", "band_lo")
        .agg(
            band_recall=pl.col(PER_POSITIVE_AVG_RECALL_COLUMN).cast(pl.Float64).mean(),
            n_band_pos=pl.len(),
            n_pos=pl.col("n_pos").first(),
        )
        .with_columns(stage=pl.lit(PER_USER_STAGE))
        .select("name", "stage", "seed", "user_id", "band", "band_lo", "band_recall", "n_band_pos", "n_pos")
        .sort("band_lo", "name", "user_id")
    )


def distance_band_recall_table(band_recall: pl.DataFrame) -> pl.DataFrame:
    """Recall of the positives in each distance band and the band's share of the headline, mean +- sd over seeds.

    Args:
        band_recall: Per-user band recall from ``distance_band_recall``.

    Returns:
        Frame of ``band, name, n_seeds, mean, sd, contribution, n_users, n_pos, share_of_pos``, ordered by band then
        ``mean`` (descending). ``mean`` averages ``band_recall`` over the users with a positive in the band, and
        ``contribution`` is the band's term of the per-user ``avg_recall`` averaged over every scored user, so a
        model's contributions sum to its headline. ``share_of_pos`` is the band's fraction of the positives.
    """
    seed_means = (
        band_recall
        .with_columns(
            n_seed_users=pl.col("user_id").n_unique().over("name", "seed"),
            weighted=pl.col("band_recall") * pl.col("n_band_pos") / pl.col("n_pos"),
        )
        .group_by("name", "seed", "band", "band_lo")
        .agg(
            seed_mean=pl.col("band_recall").mean(),
            contribution=pl.col("weighted").sum() / pl.col("n_seed_users").first(),
            n_users=pl.len(),
            n_pos=pl.col("n_band_pos").sum(),
        )
    )

    return (
        seed_means
        .group_by("name", "band", "band_lo")
        .agg(
            n_seeds=pl.len(),
            mean=pl.col("seed_mean").mean(),
            sd=pl.col("seed_mean").std(),
            contribution=pl.col("contribution").mean(),
            n_users=pl.col("n_users").mean(),
            n_pos=pl.col("n_pos").mean(),
        )
        .with_columns(share_of_pos=pl.col("n_pos") / pl.col("n_pos").sum().over("name"))
        .sort("band_lo", "mean", descending=[False, True])
        .select("band", "name", "n_seeds", "mean", "sd", "contribution", "n_users", "n_pos", "share_of_pos")
    )


def within_band_auc(
    per_user_results: pl.DataFrame,
    score_vectors: pl.DataFrame,
    home_distances: pl.DataFrame,
    bands: Sequence[float],
) -> pl.DataFrame:
    """Each user's mean AUC against the unvisited nodes of a distance band, over their positives in that band.

    Args:
        per_user_results: Per-user results frame from ``load_report_run``.
        score_vectors: Per-user score vectors from ``load_score_vectors``.
        home_distances: Per-user distances from home from ``load_home_distances``.
        bands: Upper band edges in km.

    Returns:
        Frame of ``name, stage, seed, user_id, band, band_lo, auc, n_band_pos, n_band_neg``, one row per model, seed,
        user and band the user has a positive in. Positives whose band holds no unvisited node are dropped.
    """
    lower_edges, upper_edges = _band_edges(bands)
    distance_by_user = dict(home_distances.select("user_id", "distances").iter_rows())
    per_user = per_user_results.filter(pl.col("stage") == PER_USER_STAGE)

    rows = []

    for name, seed in score_vectors.select("name", "seed").unique().sort("name", "seed").iter_rows():
        model_rows = _model_seed_rows(per_user, name, seed).join(
            _model_seed_rows(score_vectors, name, seed).select("user_id", "scores"), on="user_id"
        )
        user_ids = model_rows["user_id"].to_list()
        band_index = distance_band_index(
            np.asarray([distance_by_user[user] for user in user_ids], dtype=np.float64), bands
        )
        positives = model_rows[PER_POSITIVE_NODE_COLUMN].to_list()

        auc, n_negatives = _per_positive_band_auc(
            np.asarray(model_rows["scores"].to_list(), dtype=np.float64),
            positives,
            model_rows[HOME_NODE_COLUMN].to_numpy(),
            band_index,
        )
        positive_user = np.repeat(np.arange(len(user_ids)), [len(nodes) for nodes in positives])
        positive_band = band_index[positive_user, np.concatenate(positives).astype(int)]

        rows.append(
            pl.DataFrame({
                "name": pl.Series([name] * len(auc), dtype=pl.String),
                "seed": pl.Series([seed] * len(auc), dtype=pl.Int64),
                "user_id": np.asarray(user_ids)[positive_user],
                "band_lo": lower_edges[positive_band],
                "band_hi": upper_edges[positive_band],
                "auc": auc,
                "n_band_neg": n_negatives,
            })
        )

    return (
        pl
        .concat(rows)
        .filter(pl.col("auc").is_not_nan())
        .with_columns(band=_distance_band_label(pl.col("band_lo"), pl.col("band_hi")))
        .group_by("name", "seed", "user_id", "band", "band_lo")
        .agg(auc=pl.col("auc").mean(), n_band_pos=pl.len(), n_band_neg=pl.col("n_band_neg").mean())
        .with_columns(stage=pl.lit(PER_USER_STAGE))
        .select("name", "stage", "seed", "user_id", "band", "band_lo", "auc", "n_band_pos", "n_band_neg")
        .sort("band_lo", "name", "user_id")
    )


def _model_seed_rows(results: pl.DataFrame, name: str, seed: int | None) -> pl.DataFrame:
    """Rows of one model and seed, the seedless rows being the baselines'."""
    rows = results.filter(pl.col("name") == name)

    return rows.filter(pl.col("seed").is_null() if seed is None else pl.col("seed") == seed)


def within_band_auc_table(band_auc: pl.DataFrame) -> pl.DataFrame:
    """Within-band AUC of each model in each distance band, mean +- sd over training seeds.

    Args:
        band_auc: Per-user within-band AUC from ``within_band_auc``.

    Returns:
        Frame of ``band, name, n_seeds, mean, sd, n_users, n_pos, n_unvisited``, ordered by band then ``mean``
        (descending). ``n_users`` and ``n_pos`` are the users scored in the band and their positives in it, and
        ``n_unvisited`` the unvisited nodes each positive is ranked against, all averaged over seeds.
    """
    seed_means = band_auc.group_by("name", "seed", "band", "band_lo").agg(
        seed_mean=pl.col("auc").mean(),
        n_users=pl.len(),
        n_pos=pl.col("n_band_pos").sum(),
        n_unvisited=pl.col("n_band_neg").mean(),
    )

    return (
        seed_means
        .group_by("name", "band", "band_lo")
        .agg(
            n_seeds=pl.len(),
            mean=pl.col("seed_mean").mean(),
            sd=pl.col("seed_mean").std(),
            n_users=pl.col("n_users").mean(),
            n_pos=pl.col("n_pos").mean(),
            n_unvisited=pl.col("n_unvisited").mean(),
        )
        .sort("band_lo", "mean", descending=[False, True])
        .select("band", "name", "n_seeds", "mean", "sd", "n_users", "n_pos", "n_unvisited")
    )


def paired_within_band_auc(
    band_auc: pl.DataFrame,
    reference_model: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """``paired_comparison`` run on each distance band's within-band AUC.

    Args:
        band_auc: Per-user within-band AUC from ``within_band_auc``.
        reference_model: Model every other model is compared against.
        n_bootstrap: Bootstrap resamples behind the confidence interval.
        seed: Seed of the bootstrap.

    Returns:
        Frame of ``band, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``, ordered by band then
        ``mean_diff`` (descending).
    """
    return _paired_by_distance_band(band_auc, reference_model, "auc", n_bootstrap, seed)


def paired_distance_band_recall(
    band_recall: pl.DataFrame,
    reference_model: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pl.DataFrame:
    """``paired_comparison`` run on each distance band's ``band_recall``.

    Args:
        band_recall: Per-user band recall from ``distance_band_recall``.
        reference_model: Model every other model is compared against.
        n_bootstrap: Bootstrap resamples behind the confidence interval.
        seed: Seed of the bootstrap.

    Returns:
        Frame of ``band, name, n_users, mean_diff, ci_lo, ci_hi, wilcoxon_p, per_seed_sd``, ordered by band then
        ``mean_diff`` (descending).
    """
    return _paired_by_distance_band(band_recall, reference_model, "band_recall", n_bootstrap, seed)


def _paired_by_distance_band(
    band_frame: pl.DataFrame, reference_model: str, value: str, n_bootstrap: int, seed: int
) -> pl.DataFrame:
    """``paired_comparison`` of the per-user ``value`` column run on each distance band of ``band_frame``."""
    bands = band_frame.select("band", "band_lo").unique().sort("band_lo")["band"]

    comparisons = [
        paired_comparison(
            band_frame.filter(pl.col("band") == band), reference_model, value, n_bootstrap, seed
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

    positives = (
        rows
        .select("row", "name", "stage", "seed", "user_id", N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
        .explode(N_SCORED_HIGHER_COLUMN, N_TIED_COLUMN)
        .join(cutoffs, how="cross")
    )
    p_in_top_k = expected_recall_at_k(
        positives[N_SCORED_HIGHER_COLUMN].to_numpy(), positives[N_TIED_COLUMN].to_numpy(), positives["k"].to_numpy()
    )

    return (
        positives
        .with_columns(p_in_top_k=pl.Series(p_in_top_k, dtype=pl.Float64))
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


def lift_over_reference(per_user_results: pl.DataFrame, reference_model: str, ks: Sequence[int]) -> pl.DataFrame:
    """``recall_curve`` with each model's mean recall@k divided by that of ``reference_model`` at the same k.

    Returns:
        Frame of ``name, k, n_seeds, mean, sd, lift``, ordered by k then mean (descending).

    Raises:
        KeyError: If ``reference_model`` is not in the per-user frame.
    """
    curve = recall_curve(per_user_results, ks)
    reference = curve.filter(pl.col("name") == reference_model).select("k", reference_mean="mean")

    if reference.is_empty():
        raise KeyError(f"Reference model {reference_model!r} not found in the per-user frame")

    return (
        curve
        .join(reference, on="k")
        .with_columns(lift=pl.col("mean") / pl.col("reference_mean"))
        .drop("reference_mean")
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
    max_recall_k: int,
    distance_bands: Sequence[float],
    run: int | None = None,
    analysis: AnalysisConfig | None = None,
) -> None:
    """Print results analyses for a run of the whole comparison experiment: aggregate results, paired comparison,
    hop bands and distance bands.

    Args:
        report_data_path: path to the Parquet results tables (e.g. ``reports/data/``).
        name: Dataset name used in filename, e.g. ``"GenevaTPG"``.
        max_recall_k: Largest rank cutoff ``K`` of ``avg_recall@K``.
        distance_bands: Upper edges in km of the distance bands, for the band split and the within-band AUC.
        run: Run number ``n`` of ``{name}-results-{n}.parquet``. Uses latest available if None.
        analysis: Analysis configuration. Uses ``DEFAULT_ANALYSIS`` if None. Its ``extra_runs`` add the baselines of
            other runs of the same dataset.
    """
    report_data_path: Path = Path(report_data_path)
    run = run if run is not None else _latest_run(report_data_path, name)
    analysis = DEFAULT_ANALYSIS if analysis is None else analysis
    aggregate, per_user = load_report_run(report_data_path, name, run, max_recall_k)

    n_scored, n_dropped = _num_scored_dropped_users(aggregate)
    metric = analysis.per_user_metric
    reference = analysis.reference_model
    ks = analysis.recall_curve_ks

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(f"\n=== {name} run {run} ===")
        print(f"test users: {n_scored} scored, {n_dropped} dropped due to empty RG_i")

        print(f"\n-- per-user {metric} (K = {max_recall_k}), avg over seeds --")
        print(per_user_metric_summary(per_user, metric))

        print(f"\n-- per-user recall@k at k = {ks}, avg over seeds, lift over {POPULARITY_MODEL} --")
        print(lift_over_reference(per_user, POPULARITY_MODEL, ks))

        print(f"\n-- paired per-user recall@k at k = {ks}, vs {reference} --")
        print(paired_comparison_by_k(per_user, reference, ks))

        print(f"\n-- {R_PRECISION_COLUMN}, avg over seeds (diagnostic) --")
        print(aggregate_metrics(aggregate, R_PRECISION_COLUMN))

        print(f"\n-- paired per-user {metric}, vs {reference} (avg over seeds) --")
        print(paired_comparison(per_user, reference, metric))

        min_size = analysis.min_realised_size
        restricted = _restrict_to_realised_size(per_user, min_size)

        print("\n-- test users by |RG_i| --")
        print(realised_size_summary(per_user, n_dropped))

        print(f"\n-- per-user {metric} where |RG_i| >= {min_size}, avg over seeds --")
        print(per_user_metric_summary(restricted, metric))

        print(f"\n-- paired per-user {metric} where |RG_i| >= {min_size}, vs {reference} --")
        print(paired_comparison(restricted, reference, metric))

        print(f"\n-- per-user {metric} by hop band of each positive, avg over seeds --")
        print(band_decomposition(per_user, metric))

        print(f"\n-- paired per-user {metric} by hop band of each positive, vs {reference} --")
        print(paired_comparison_by_band(per_user, reference, metric))

        by_distance = distance_band_recall(per_user, load_home_distances(report_data_path, name, run), distance_bands)

        print(f"\n-- per-user {metric} by distance band of each positive, edges {distance_bands} km, avg over seeds --")
        print(distance_band_recall_table(by_distance))

        print(f"\n-- paired per-user {metric} by distance band of each positive, vs {reference} --")
        print(paired_distance_band_recall(by_distance, reference))

        print("\n-- home coverage of test users --")
        print(home_coverage_summary(per_user))

        print(f"\n-- paired per-user {metric}, vs {reference}, by home coverage --")
        print(paired_comparison_by_home_coverage(per_user, reference, metric))

        if (report_data_path / f"{name}-scores-{run}.parquet").exists():
            scores = load_score_vectors(report_data_path, name, run)
            home_distances = load_home_distances(report_data_path, name, run)
            band_auc = within_band_auc(per_user, scores, home_distances, distance_bands)

            print(f"\n-- AUC within each positive's distance band, edges {distance_bands} km, avg over seeds --")
            print(within_band_auc_table(band_auc))

            print(f"\n-- paired AUC within each positive's distance band, vs {reference} --")
            print(paired_within_band_auc(band_auc, reference))

            print("\n-- model health checks --")
            print(check_model_health(scores, per_user, POPULARITY_MODEL))
