"""Tests for the post-hoc analysis of experiment outputs (seed spread, pairing, hop bands, home zones)."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import polars as pl
import pytest
import torch
import torch_geometric as pyg

from activitygraphs.analysis import (
    DROPPED_USERS_COLUMN,
    REALISED_SIZE_COLUMN,
    SCORED_USERS_COLUMN,
    _average_metric_over_seeds,
    _diff_sd_per_seed,
    _latest_run,
    aggregate_metrics,
    append_baselines,
    band_contributions,
    band_decomposition,
    paired_comparison_by_band,
    distance_band_recall,
    distance_band_recall_table,
    paired_distance_band_recall,
    paired_within_band_auc,
    within_band_auc,
    within_band_auc_table,
    paired_comparison_by_k,
    recall_at_ks,
    recall_curve,
    home_zone_summary,
    lift_over_reference,
    with_ranking_metrics,
    check_home_node_rank,
    load_run,
    load_score_vectors,
    check_model_health,
    check_overfit_health,
    paired_comparison,
    check_popularity_correlation,
    check_user_invariance,
    per_user_metric_summary,
    realised_size_summary,
    _restrict_to_realised_size,
    _num_scored_dropped_users,
)
from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.models import NodeMLP
from activitygraphs.ml.training import PER_USER_STAGE, SCORE_VECTOR_STAGE

MAIN_METRIC = "test_r_precision"
PER_USER_METRIC = "r_precision"
BUDGET = 50


def make_aggregate() -> pl.DataFrame:
    """Two seeds of one model plus a deterministic baseline."""
    rows = [
        {"name": "MLP", "stage": "test", "seed": 42, "test_r_precision": 0.40},
        {"name": "MLP", "stage": "test", "seed": 43, "test_r_precision": 0.50},
        {"name": "Baseline", "stage": "test", "seed": None, "test_r_precision": 0.30},
        {"name": "MLP", "stage": "fit", "seed": 42, "test_r_precision": None},
    ]

    return pl.DataFrame(rows).with_columns(**{SCORED_USERS_COLUMN: pl.lit(90.0), DROPPED_USERS_COLUMN: pl.lit(10.0)})


# Home-included |RG_i| of users 1, 2 and 3: only the last two clear the `min_realised_size` of 3.
REALISED_SIZES = {1: 2, 2: 3, 3: 4}


def make_per_user() -> pl.DataFrame:
    """Three users scored by a model at two seeds and by a baseline once; the model wins on every user."""
    rows = [
        {"name": name, "stage": PER_USER_STAGE, "user_id": user, "seed": seed, "r_precision": score}
        for name, seed, scores in [("MLP", 42, [0.5, 0.5, 1.0]), ("MLP", 43, [0.5, 1.0, 1.0])]
        for user, score in zip([1, 2, 3], scores, strict=True)
    ]
    rows += [
        {"name": "Baseline", "stage": PER_USER_STAGE, "user_id": user, "seed": None, "r_precision": score}
        for user, score in zip([1, 2, 3], [0.0, 0.5, 0.5], strict=True)
    ]

    return pl.DataFrame(rows).with_columns(pl.col("user_id").replace_strict(REALISED_SIZES).alias(REALISED_SIZE_COLUMN))


@dataclass
class FakeDataset:
    """Minimal stand-in exposing only what ``home_zone_summary`` reads off an ``ActivityDataset``."""

    spatial_features: torch.Tensor
    num_nodes: int
    is_home_spatial_idx: int


def make_dataset(home_nodes: list[int], n_nodes: int = 3, is_home_idx: int = 1) -> FakeDataset:
    """Dataset whose ``is_home`` column peaks at ``home_nodes[i]`` for user ``i``."""
    spatial = torch.zeros(len(home_nodes), n_nodes, is_home_idx + 1)
    for user, node in enumerate(home_nodes):
        spatial[user, node, is_home_idx] = 1.0

    return FakeDataset(spatial_features=spatial, num_nodes=n_nodes, is_home_spatial_idx=is_home_idx)


def make_score_vectors(run_scores: list[list[float]]) -> pl.DataFrame:
    """Score-vector rows as ``save_results`` writes them to ``<name>-scores-<n>.parquet``."""
    return pl.DataFrame({
        "user_id": list(range(len(run_scores))),
        "scores": run_scores,
    }).with_columns(
        name=pl.lit("MLP-dist"),
        stage=pl.lit(SCORE_VECTOR_STAGE),
        seed=pl.lit(42, dtype=pl.Int64),
    )


class TestLoading:
    def test_latest_run_picks_the_highest_number(self, tmp_path):
        for run in [1, 2, 10]:
            (tmp_path / f"GenevaTPG-results-{run}.parquet").touch()
        (tmp_path / "OtherCity-results-99.parquet").touch()

        assert _latest_run(tmp_path, "GenevaTPG") == 10

    def test_latest_run_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _latest_run(tmp_path, "GenevaTPG")

    def test_load_run_reads_both_frames_of_the_latest_run(self, tmp_path):
        make_aggregate().write_parquet(tmp_path / "GenevaTPG-results-1.parquet")
        make_per_user().write_parquet(tmp_path / "GenevaTPG-per-user-1.parquet")
        make_aggregate().head(1).write_parquet(tmp_path / "GenevaTPG-results-2.parquet")
        make_per_user().head(1).write_parquet(tmp_path / "GenevaTPG-per-user-2.parquet")

        aggregate, per_user = load_run(tmp_path, "GenevaTPG")
        assert aggregate.height == 1
        assert per_user.height == 1

        aggregate, per_user = load_run(tmp_path, "GenevaTPG", run=1)
        assert aggregate.height == 4
        assert per_user.height == 9


class TestLoadScoreVectors:
    def test_reads_the_score_frame_of_the_latest_run(self, tmp_path):
        make_aggregate().write_parquet(tmp_path / "GenevaTPG-results-1.parquet")
        make_score_vectors([[0.1, 0.9]]).write_parquet(tmp_path / "GenevaTPG-scores-1.parquet")
        make_aggregate().write_parquet(tmp_path / "GenevaTPG-results-2.parquet")
        make_score_vectors([[0.2, 0.8], [0.3, 0.7]]).write_parquet(tmp_path / "GenevaTPG-scores-2.parquet")

        assert load_score_vectors(tmp_path, "GenevaTPG").height == 2
        assert load_score_vectors(tmp_path, "GenevaTPG", run=1)["scores"].to_list() == [[0.1, 0.9]]

    def test_raises_when_the_run_saved_no_score_vectors(self, tmp_path):
        make_aggregate().write_parquet(tmp_path / "GenevaTPG-results-1.parquet")

        with pytest.raises(FileNotFoundError, match="save_score_vectors"):
            load_score_vectors(tmp_path, "GenevaTPG")


class TestUserInvariance:
    def test_identical_rankings_score_one(self):
        """The THATS collapse: every user ranked identically, so the median pairwise Spearman is 1."""
        scores = np.tile(np.array([0.1, 0.9, 0.4, 0.7]), (5, 1))

        assert check_user_invariance(scores) == pytest.approx(1.0)

    def test_opposed_rankings_score_minus_one(self):
        scores = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])

        assert check_user_invariance(scores) == pytest.approx(-1.0)

    def test_a_shifted_ranking_still_counts_as_invariant(self):
        """Spearman is rank-based, so a per-user offset or scale must not register as user variation."""
        base = np.array([0.1, 0.9, 0.4, 0.7])
        scores = np.stack([base, base + 5.0, base * 3.0])

        assert check_user_invariance(scores) == pytest.approx(1.0)

    def test_independent_users_score_near_zero(self):
        scores = np.random.default_rng(0).random((60, 200))

        assert abs(check_user_invariance(scores)) < 0.1

    def test_subsamples_large_user_sets_deterministically(self):
        scores = np.random.default_rng(0).random((500, 50))

        repeated = check_user_invariance(scores, max_users=50, seed=1)

        assert check_user_invariance(scores, max_users=50, seed=1) == repeated

    def test_a_single_user_has_no_pairs(self):
        assert math.isnan(check_user_invariance(np.array([[0.1, 0.2]])))


class TestPopularityCorrelation:
    def test_a_popularity_clone_scores_one(self):
        popularity = np.array([0.05, 0.6, 0.2, 0.9])
        scores = np.stack([popularity, popularity])

        assert check_popularity_correlation(scores, popularity) == pytest.approx(1.0)

    def test_monotone_transforms_of_popularity_also_score_one(self):
        """The reference is used as an ordering, so logit(p_n) and p_n must be interchangeable."""
        popularity = np.array([0.05, 0.6, 0.2, 0.9])
        logits = np.log(popularity / (1 - popularity))

        assert check_popularity_correlation(np.stack([popularity]), logits) == pytest.approx(1.0)

    def test_a_user_specific_model_averages_away_from_popularity(self):
        popularity = np.array([0.9, 0.6, 0.3, 0.1])
        scores = np.array([[0.1, 0.2, 0.3, 0.9], [0.2, 0.1, 0.9, 0.3]])

        assert check_popularity_correlation(scores, popularity) < 0.0


class TestHomeNodeRank:
    def test_home_at_the_top_for_everyone(self):
        scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])

        assert check_home_node_rank(scores, np.array([0, 1])) == (1.0, 1.0)

    def test_home_at_the_bottom(self):
        scores = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])

        is_top1, median_rank = check_home_node_rank(scores, np.array([0, 0]))
        assert (is_top1, median_rank) == (0.0, 3.0)

    def test_ranks_count_strictly_higher_scores(self):
        """A flat field ranks home first; the user-invariance check is what catches that case, not this one."""
        scores = np.ones((2, 4))

        assert check_home_node_rank(scores, np.array([2, 3])) == (1.0, 1.0)


class TestOverfitHealth:
    def _fit_rows(self) -> pl.DataFrame:
        return pl.DataFrame([
            {"name": "MLP-dist", "stage": "fit", "epoch": 0, "train_avg_recall@50": 0.3},
            {"name": "MLP-dist", "stage": "fit", "epoch": 1, "train_avg_recall@50": 0.99},
            {"name": "MLP-dist", "stage": "fit", "epoch": 2, "train_avg_recall@50": 0.97},
            {"name": "GATSkip", "stage": "fit", "epoch": 0, "train_avg_recall@50": 0.2},
            {"name": "GATSkip", "stage": "fit", "epoch": 1, "train_avg_recall@50": 0.25},
        ])

    def test_reports_best_and_final_per_model(self):
        health = check_overfit_health(self._fit_rows(), BUDGET)

        assert health["name"].to_list() == ["GATSkip", "MLP-dist"]  # worst first
        assert health["best_train_avg_recall"].to_list() == pytest.approx([0.25, 0.99])
        assert health["final_train_avg_recall"].to_list() == pytest.approx([0.25, 0.97])

    def test_raises_without_a_train_ranking_column(self):
        with pytest.raises(KeyError, match="log_train_ranking"):
            check_overfit_health(make_aggregate(), BUDGET)


class TestModelHealthChecks:
    """Three models over 4 nodes: a collapsed popularity clone, a home-aware model, and the baseline."""

    POPULARITY = [0.4, 0.3, 0.2, 0.1]
    HOMES = {0: 3, 1: 2}

    def _score_vectors(self) -> pl.DataFrame:
        rows = []
        for seed in (42, 43):
            for user_id, home in self.HOMES.items():
                collapsed = list(self.POPULARITY)
                home_aware = list(self.POPULARITY)
                home_aware[home] = 0.99
                rows.append({"name": "Collapsed", "seed": seed, "user_id": user_id, "scores": collapsed})
                rows.append({"name": "HomeAware", "seed": seed, "user_id": user_id, "scores": home_aware})

        for user_id in self.HOMES:
            rows.append({"name": "VisitFrequency", "seed": None, "user_id": user_id, "scores": list(self.POPULARITY)})

        return pl.DataFrame(rows, schema_overrides={"seed": pl.Int64})

    def _per_user(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": list(self.HOMES),
            "home_node": list(self.HOMES.values()),
            "stage": [PER_USER_STAGE] * len(self.HOMES),
        })

    def test_separates_a_collapsed_model_from_a_home_aware_one(self):
        health = check_model_health(self._score_vectors(), self._per_user(), "VisitFrequency")
        by_name = {row["name"]: row for row in health.to_dicts()}

        # The collapsed model ranks every user identically and is exactly the popularity ordering.
        assert by_name["Collapsed"]["user_invariance"] == pytest.approx(1.0)
        assert by_name["Collapsed"]["popularity_corr"] == pytest.approx(1.0)
        assert by_name["Collapsed"]["home_is_top1"] == pytest.approx(0.0)

        # The home-aware model ranks each user's own home first, so its users disagree with each other.
        assert by_name["HomeAware"]["home_is_top1"] == pytest.approx(1.0)
        assert by_name["HomeAware"]["home_median_rank"] == pytest.approx(1.0)
        assert by_name["HomeAware"]["user_invariance"] < 1.0

    def test_averages_over_seeds_and_counts_them(self):
        health = check_model_health(self._score_vectors(), self._per_user(), "VisitFrequency")
        by_name = {row["name"]: row for row in health.to_dicts()}

        assert by_name["HomeAware"]["n_seeds"] == 2
        assert by_name["VisitFrequency"]["n_seeds"] == 1  # deterministic, no seed
        assert by_name["HomeAware"]["home_is_top1_sd"] == pytest.approx(0.0)

    def test_the_collapsed_models_sort_above_the_home_aware_one(self):
        """`VisitFrequency` is collapsed by construction, so it ties `Collapsed` at the top."""
        health = check_model_health(self._score_vectors(), self._per_user(), "VisitFrequency")

        assert health["name"].to_list() == ["Collapsed", "VisitFrequency", "HomeAware"]

    def test_baseline_reproduces_the_popularity_ordering(self):
        """VisitFrequency is the popularity reference, so it must score 1.0 against itself - a control on the check."""
        health = check_model_health(self._score_vectors(), self._per_user(), "VisitFrequency")
        by_name = {row["name"]: row for row in health.to_dicts()}

        assert by_name["VisitFrequency"]["popularity_corr"] == pytest.approx(1.0)

    def test_raises_when_the_popularity_reference_is_missing(self):
        scores = self._score_vectors().filter(pl.col("name") != "VisitFrequency")

        with pytest.raises(KeyError, match="VisitFrequency"):
            check_model_health(scores, self._per_user(), "VisitFrequency")


class TestSeedSummary:
    def test_reports_mean_and_sd_over_seeds(self):
        row = aggregate_metrics(make_aggregate(), MAIN_METRIC).filter(pl.col("name") == "MLP").to_dicts()[0]
        assert row["n_seeds"] == 2
        assert row["mean"] == 0.45
        assert abs(row["sd"] - 0.0707106) < 1e-6  # sample sd of {0.40, 0.50}

    def test_deterministic_baseline_has_one_seed_and_no_sd(self):
        row = aggregate_metrics(make_aggregate(), MAIN_METRIC).filter(pl.col("name") == "Baseline").to_dicts()[0]
        assert row["n_seeds"] == 1
        assert row["sd"] is None

    def test_ignores_fit_rows(self):
        aggregate = make_aggregate().with_columns(
            test_r_precision=pl.when(pl.col("stage") == "fit").then(99.0).otherwise("test_r_precision")
        )
        assert aggregate_metrics(aggregate, MAIN_METRIC)["mean"].max() < 1.0

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            aggregate_metrics(make_aggregate(), "test_not_a_metric")


class TestRealisedSizeRestriction:
    """With home out of ``RG_i``, a user below ``|RG_i| >= 3`` keeps at most one non-home node."""

    def test_users_below_the_threshold_are_dropped(self):
        restricted = _restrict_to_realised_size(make_per_user(), min_size=3)

        assert sorted(restricted["user_id"].unique().to_list()) == [2, 3]

    def test_non_per_user_rows_pass_through(self):
        """Aggregate and score-vector rows have no set size of their own and must survive the filter."""
        per_user = make_per_user()
        other = per_user.head(1).with_columns(
            stage=pl.lit(SCORE_VECTOR_STAGE),
            **{REALISED_SIZE_COLUMN: pl.lit(None, dtype=pl.Int64)},
        )

        restricted = _restrict_to_realised_size(pl.concat([per_user, other]), min_size=3)

        assert restricted.filter(pl.col("stage") == SCORE_VECTOR_STAGE).height == 1

    def test_summary_counts_each_user_once_across_models_and_seeds(self):
        summary = realised_size_summary(make_per_user(), n_dropped=0)

        assert summary[REALISED_SIZE_COLUMN].to_list() == [2, 3, 4]
        assert summary["n_users"].to_list() == [1, 1, 1]
        assert summary["share_of_users"].to_list() == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_dropped_users_enter_the_denominator(self):
        """Users the exclusion left with nothing to rank are absent from the per-user frame."""
        summary = realised_size_summary(make_per_user(), n_dropped=1)

        assert summary["n_users"].to_list() == [1, 1, 1, 1]
        assert summary["scored"].to_list() == [False, True, True, True]
        # Every share is now a share of the whole test split, not of the part that survived.
        assert summary["share_of_users"].to_list() == pytest.approx([0.25] * 4)

    def test_the_dropped_row_is_reported_at_a_home_included_size_of_one(self):
        summary = realised_size_summary(make_per_user(), n_dropped=4)
        dropped = summary.filter(~pl.col("scored")).to_dicts()[0]

        assert dropped[REALISED_SIZE_COLUMN] == 1
        assert dropped["n_users"] == 4

    def test_no_dropped_row_when_the_count_is_zero(self):
        summary = realised_size_summary(make_per_user(), n_dropped=0)

        assert summary["scored"].to_list() == [True, True, True]


class TestUserPopulation:
    """The headline covers fewer users than the sanity metric; the frame has to say how many fewer."""

    def _aggregate(self, scored: int = 90, dropped: int = 10) -> pl.DataFrame:
        return make_aggregate().with_columns(**{
            SCORED_USERS_COLUMN: pl.lit(float(scored)),
            DROPPED_USERS_COLUMN: pl.lit(float(dropped)),
        })

    def test_reads_both_counts(self):
        assert _num_scored_dropped_users(self._aggregate()) == (90, 10)

    def test_the_restriction_changes_the_paired_comparison(self):
        per_user = make_per_user()

        full = paired_comparison(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200).to_dicts()[0]
        restricted = paired_comparison(
            _restrict_to_realised_size(per_user, min_size=3), "Baseline", PER_USER_METRIC, n_bootstrap=200
        ).to_dicts()[0]

        assert full["n_users"] == 3
        assert restricted["n_users"] == 2
        # User 1 carried the largest gap (0.5 vs 0.0), so dropping it shrinks the mean difference.
        assert restricted["mean_diff"] < full["mean_diff"]


class TestPerUserMetricSummary:
    def test_reports_mean_and_sd_over_seeds(self):
        row = per_user_metric_summary(make_per_user(), PER_USER_METRIC).filter(pl.col("name") == "MLP").to_dicts()[0]

        assert row["n_seeds"] == 2
        assert row["n_users"] == 3
        # Seed means of {0.5, 0.5, 1.0} and {0.5, 1.0, 1.0}.
        assert row["mean"] == pytest.approx((2 / 3 + 5 / 6) / 2)

    def test_deterministic_baseline_has_one_seed_and_no_sd(self):
        row = (
            per_user_metric_summary(make_per_user(), PER_USER_METRIC).filter(pl.col("name") == "Baseline").to_dicts()[0]
        )

        assert row["n_seeds"] == 1
        assert row["sd"] is None


class TestPairedComparison:
    def test_averages_each_user_over_seeds(self):
        scores = _average_metric_over_seeds(make_per_user(), PER_USER_METRIC).filter(pl.col("name") == "MLP")
        assert scores["r_precision"].to_list() == [0.5, 0.75, 1.0]

    def test_mean_difference_is_the_seed_averaged_paired_mean(self):
        row = paired_comparison(make_per_user(), "Baseline", PER_USER_METRIC, n_bootstrap=200).to_dicts()[0]
        assert row["n_users"] == 3  # each user counted once, not once per seed
        assert abs(row["mean_diff"] - 1.25 / 3) < 1e-9  # mean of {0.5, 0.25, 0.5}

    def test_interval_brackets_the_mean_difference(self):
        row = paired_comparison(make_per_user(), "Baseline", PER_USER_METRIC, n_bootstrap=500).to_dicts()[0]
        assert row["ci_lo"] <= row["mean_diff"] <= row["ci_hi"]

    def test_reference_is_excluded_from_the_comparison(self):
        compared = paired_comparison(make_per_user(), "Baseline", PER_USER_METRIC, n_bootstrap=200)
        assert compared["name"].to_list() == ["MLP"]

    def test_identical_scores_give_a_zero_difference(self):
        per_user = make_per_user().with_columns(r_precision=pl.lit(0.5))
        row = paired_comparison(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200).to_dicts()[0]
        assert row["mean_diff"] == 0.0
        assert row["wilcoxon_p"] == 1.0

    def test_missing_reference_raises(self):
        with pytest.raises(KeyError):
            paired_comparison(make_per_user(), "NotAModel", PER_USER_METRIC, n_bootstrap=200)


class TestDiffSdPerSeed:
    def test_seedless_reference_is_reused_for_every_seed(self):
        # Per-seed mean diffs against the baseline are 1.0/3 (seed 42) and 1.5/3 (seed 43).
        sd = _diff_sd_per_seed(make_per_user(), "MLP", "Baseline", PER_USER_METRIC)
        assert abs(sd - pl.Series([1.0 / 3, 1.5 / 3]).std()) < 1e-12

    def test_seeded_reference_is_paired_seed_for_seed(self):
        # A reference with its own seeds: seed 42 is 0.1 below the model on every user, seed 43 is 0.3 below.
        reference = (
            make_per_user()
            .filter(pl.col("name") == "MLP")
            .with_columns(
                name=pl.lit("MLP-ref"),
                r_precision=pl.col("r_precision") - pl.when(pl.col("seed") == 42).then(0.1).otherwise(0.3),
            )
        )
        per_user = pl.concat([make_per_user(), reference])

        sd = _diff_sd_per_seed(per_user, "MLP", "MLP-ref", PER_USER_METRIC)
        assert abs(sd - pl.Series([0.1, 0.3]).std()) < 1e-12

    def test_seeded_reference_carries_more_spread_than_its_seed_average(self):
        # A seedless reference holding the same per-user scores as the seeded one averaged over its
        # seeds hides the reference's own run-to-run spread: the per-seed gaps are then 0.117 and
        # 0.283 (the model's spread alone) rather than the true 0.1 and 0.3.
        seeded_reference = (
            make_per_user()
            .filter(pl.col("name") == "MLP")
            .with_columns(
                name=pl.lit("MLP-ref"),
                r_precision=pl.col("r_precision") - pl.when(pl.col("seed") == 42).then(0.1).otherwise(0.3),
            )
        )
        averaged_reference = (
            seeded_reference
            .group_by("name", "stage", "user_id")
            .agg(pl.col("r_precision").mean())
            .with_columns(seed=pl.lit(None, dtype=pl.Int64))
        )

        paired = _diff_sd_per_seed(pl.concat([make_per_user(), seeded_reference]), "MLP", "MLP-ref", PER_USER_METRIC)
        averaged = _diff_sd_per_seed(
            pl.concat([make_per_user(), averaged_reference], how="diagonal"), "MLP", "MLP-ref", PER_USER_METRIC
        )

        assert paired == pytest.approx(pl.Series([0.1, 0.3]).std())
        assert averaged == pytest.approx(pl.Series([2 / 3 - 0.55, 5 / 6 - 0.55]).std())
        assert paired > averaged

    def test_single_seed_gives_nan_not_none(self):
        per_user = make_per_user().filter(pl.col("seed") != 43)
        assert math.isnan(_diff_sd_per_seed(per_user, "MLP", "Baseline", PER_USER_METRIC))

    def test_reported_alongside_the_paired_comparison(self):
        row = paired_comparison(make_per_user(), "Baseline", PER_USER_METRIC, n_bootstrap=200).to_dicts()[0]
        assert row["per_seed_sd"] == pytest.approx(
            _diff_sd_per_seed(make_per_user(), "MLP", "Baseline", PER_USER_METRIC)
        )


def make_positive_per_user() -> pl.DataFrame:
    """Two users scored by a model at two seeds and by a baseline once, with each positive's hop distance and flags."""
    user_hops = {1: [1.0, float("inf")], 2: [3.0, 5.0]}
    top_r_flags = [
        ("MLP", 42, {1: [True, True], 2: [False, False]}),
        ("MLP", 43, {1: [True, False], 2: [True, False]}),
        ("Baseline", None, {1: [True, False], 2: [False, False]}),
    ]
    scored_higher = {"MLP": {1: [0, 1], 2: [2, 3]}, "Baseline": {1: [0, 10], 2: [60, 49]}}
    rows = [
        {
            "name": name,
            "stage": PER_USER_STAGE,
            "seed": seed,
            "user_id": user,
            "n_pos": len(hops),
            "r_precision": sum(flags[user]) / len(hops),
            "pos_hops": hops,
            "pos_in_top_r": flags[user],
            "pos_n_scored_higher": scored_higher[name][user],
            "pos_n_tied": [0] * len(hops),
        }
        for name, seed, flags in top_r_flags
        for user, hops in user_hops.items()
    ]

    return pl.DataFrame(rows)


class TestBandDecomposition:
    def test_every_user_gets_a_row_in_every_band(self):
        contributions = band_contributions(make_positive_per_user(), PER_USER_METRIC)

        assert contributions.height == 6 * 3
        user_2_far = contributions.filter((pl.col("user_id") == 2) & (pl.col("band") == "13+"))
        assert user_2_far["contribution"].to_list() == [0.0, 0.0, 0.0]

    def test_contributions_sum_to_the_per_user_metric(self):
        per_user = make_positive_per_user().with_columns(pl.col("seed").fill_null(-1))
        summed = (
            band_contributions(per_user, PER_USER_METRIC)
            .group_by("name", "seed", "user_id")
            .agg(pl.col("contribution").sum())
            .join(per_user.select("name", "seed", "user_id", "r_precision"), on=["name", "seed", "user_id"])
        )

        assert summed["contribution"].to_list() == pytest.approx(summed["r_precision"].to_list())

    def test_band_means_sum_to_the_headline(self):
        per_user = make_positive_per_user()
        table = band_decomposition(per_user, PER_USER_METRIC).filter(pl.col("name") == "MLP")
        headline = per_user_metric_summary(per_user, PER_USER_METRIC).filter(pl.col("name") == "MLP")["mean"][0]

        means = dict(zip(table["band"], table["mean"], strict=True))
        assert means == pytest.approx({"0-2": 0.25, "3-5": 0.125, "13+": 0.125})
        assert sum(means.values()) == pytest.approx(headline)

    def test_shares_of_the_metric_and_of_the_positives(self):
        table = band_decomposition(make_positive_per_user(), PER_USER_METRIC).filter(pl.col("name") == "MLP")

        assert dict(zip(table["band"], table["share_of_metric"], strict=True)) == pytest.approx({
            "0-2": 0.5,
            "3-5": 0.25,
            "13+": 0.25,
        })
        assert dict(zip(table["band"], table["share_of_pos"], strict=True)) == pytest.approx({
            "0-2": 0.25,
            "3-5": 0.5,
            "13+": 0.25,
        })

    def test_bands_are_ordered_by_their_lower_bound(self):
        table = band_decomposition(make_positive_per_user(), PER_USER_METRIC).filter(pl.col("name") == "MLP")
        assert table["band"].to_list() == ["0-2", "3-5", "13+"]

    def test_a_metric_without_a_per_positive_value_raises(self):
        with pytest.raises(KeyError, match="decomposable metrics"):
            band_contributions(make_positive_per_user(), "ndcg")

    def test_average_recall_contributions_sum_to_the_derived_per_user_value(self):
        per_user = with_ranking_metrics(make_positive_per_user(), BUDGET).with_columns(pl.col("seed").fill_null(-1))
        summed = (
            band_contributions(per_user, "avg_recall")
            .group_by("name", "seed", "user_id")
            .agg(pl.col("contribution").sum())
            .join(per_user.select("name", "seed", "user_id", "avg_recall"), on=["name", "seed", "user_id"])
        )

        assert summed.height == 6
        assert summed["contribution"].to_list() == pytest.approx(summed["avg_recall"].to_list())

    def test_average_recall_band_split_is_hand_computed(self):
        """Baseline user 1: the hop-1 positive at rank 1 gives 1/2, the unreachable one at rank 11 gives 0.8 / 2."""
        per_user = with_ranking_metrics(make_positive_per_user(), BUDGET)
        contributions = band_contributions(per_user, "avg_recall").filter(
            (pl.col("name") == "Baseline") & (pl.col("user_id") == 1)
        )

        by_band = dict(zip(contributions["band"], contributions["contribution"], strict=True))
        assert by_band == pytest.approx({"0-2": 0.5, "3-5": 0.0, "13+": 0.4})

    def test_paired_band_differences_sum_to_the_overall_difference(self):
        per_user = make_positive_per_user()
        by_band = paired_comparison_by_band(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200)
        overall = paired_comparison(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200)["mean_diff"][0]

        diffs = dict(zip(by_band["band"], by_band["mean_diff"], strict=True))
        assert diffs == pytest.approx({"0-2": 0.0, "3-5": 0.125, "13+": 0.125})
        assert sum(diffs.values()) == pytest.approx(overall)
        assert by_band["n_users"].to_list() == [2, 2, 2]


# Distance from home in metres of the six nodes of the within-band AUC fixture. With edges at 1 and 5 km, nodes 0-2
# fall in the first band, nodes 3-4 in the second and node 5 alone in the third.
NODE_DISTANCES = [0.0, 500.0, 800.0, 2000.0, 3000.0, 8000.0]
BAND_EDGES = [1.0, 5.0]

# Node 0 is every user's home, so it is not a candidate; user 2's only far positive sits alone in its band.
BAND_POSITIVES = {1: [1, 3], 2: [4, 5]}
BAND_SCORES = {"MLP": [0.9, 0.6, 0.2, 0.5, 0.4, 0.1], "Baseline": [0.5] * 6}


def make_band_per_user() -> pl.DataFrame:
    """Per-user rows of two users scored by a model at two seeds and by a baseline once."""
    rows = [
        {
            "name": name,
            "stage": PER_USER_STAGE,
            "seed": seed,
            "user_id": user,
            "n_pos": len(positives),
            "home_node": 0,
            "pos_node": positives,
        }
        for name, seed in [("MLP", 42), ("MLP", 43), ("Baseline", None)]
        for user, positives in BAND_POSITIVES.items()
    ]

    return pl.DataFrame(rows)


def make_band_score_vectors() -> pl.DataFrame:
    """Score vectors of the same models, identical across seeds and users."""
    rows = [
        {"name": name, "stage": SCORE_VECTOR_STAGE, "seed": seed, "user_id": user, "scores": BAND_SCORES[name]}
        for name, seed in [("MLP", 42), ("MLP", 43), ("Baseline", None)]
        for user in BAND_POSITIVES
    ]

    return pl.DataFrame(rows)


def make_band_home_distances() -> pl.DataFrame:
    return pl.DataFrame({"user_id": list(BAND_POSITIVES), "distances": [NODE_DISTANCES, NODE_DISTANCES]})


def band_auc_frame(bands: list[float] | None = None) -> pl.DataFrame:
    return within_band_auc(
        make_band_per_user(), make_band_score_vectors(), make_band_home_distances(), bands or BAND_EDGES
    )


class TestWithinBandAUC:
    def test_bands_are_labelled_by_their_edges_in_km(self):
        """Node 5's band holds no unvisited node, so no row carries its label."""
        assert sorted(band_auc_frame()["band"].unique().to_list()) == ["0-1", "1-5"]

    def test_the_home_node_is_not_one_of_the_band_negatives(self):
        """Home outscores the near positive, so counting it would drop that AUC from 1 to 0.5."""
        near = band_auc_frame().filter((pl.col("name") == "MLP") & (pl.col("seed") == 42) & (pl.col("band") == "0-1"))

        assert near["user_id"].to_list() == [1]
        assert near["auc"].to_list() == pytest.approx([1.0])
        assert near["n_band_neg"].to_list() == pytest.approx([1.0])

    def test_a_positive_alone_in_its_band_is_dropped(self):
        far = band_auc_frame().filter((pl.col("name") == "MLP") & (pl.col("seed") == 42) & (pl.col("user_id") == 2))

        assert far["band"].to_list() == ["1-5"]
        assert far["auc"].to_list() == pytest.approx([0.0])

    def test_wider_bands_rank_each_positive_against_more_nodes(self):
        """The edges are read at report time, so one run answers on any banding."""
        wide = band_auc_frame(bands=[5.0]).filter((pl.col("name") == "MLP") & (pl.col("seed") == 42))

        assert wide["band"].unique().to_list() == ["0-5"]
        assert dict(zip(wide["user_id"], wide["auc"], strict=True)) == pytest.approx({1: 1.0, 2: 1 / 3})

    def test_table_averages_users_then_seeds(self):
        table = within_band_auc_table(band_auc_frame()).filter(pl.col("name") == "MLP")

        assert table["band"].to_list() == ["0-1", "1-5"]
        assert dict(zip(table["band"], table["mean"], strict=True)) == pytest.approx({"0-1": 1.0, "1-5": 0.5})
        assert table["n_seeds"].to_list() == [2, 2]

    def test_paired_comparison_keeps_only_the_users_scored_in_the_band(self):
        by_band = paired_within_band_auc(band_auc_frame(), "Baseline", n_bootstrap=200)

        assert by_band["band"].to_list() == ["0-1", "1-5"]
        assert by_band["mean_diff"].to_list() == pytest.approx([0.5, 0.0])
        assert by_band["n_users"].to_list() == [1, 2]


# Each positive's ``pos_avg_recall`` in the order of ``BAND_POSITIVES``, the same at both seeds.
BAND_POSITIVE_RECALL = {"MLP": {1: [0.8, 0.4], 2: [0.6, 0.2]}, "Baseline": {1: [0.4, 0.4], 2: [0.2, 0.2]}}


def by_distance_frame() -> pl.DataFrame:
    per_user = make_band_per_user().with_columns(
        pos_avg_recall=pl.struct("name", "user_id").map_elements(
            lambda row: BAND_POSITIVE_RECALL[row["name"]][row["user_id"]], return_dtype=pl.List(pl.Float64)
        )
    )

    return distance_band_recall(per_user, make_band_home_distances(), BAND_EDGES)


class TestDistanceBandRecall:
    def test_a_positive_alone_in_its_band_is_kept(self):
        """Every positive is ranked against every node, so an empty band does not drop it."""
        assert by_distance_frame()["band"].unique(maintain_order=True).to_list() == ["0-1", "1-5", "5+"]

    def test_contributions_sum_to_the_headline(self):
        table = distance_band_recall_table(by_distance_frame()).filter(pl.col("name") == "MLP")

        assert dict(zip(table["band"], table["contribution"], strict=True)) == pytest.approx(
            {"0-1": 0.2, "1-5": 0.25, "5+": 0.05}
        )
        assert table["contribution"].sum() == pytest.approx(0.5)
        assert dict(zip(table["band"], table["mean"], strict=True)) == pytest.approx({"0-1": 0.8, "1-5": 0.5, "5+": 0.2})
        assert table["share_of_pos"].to_list() == pytest.approx([0.25, 0.5, 0.25])

    def test_paired_comparison_keeps_only_the_users_with_a_positive_in_the_band(self):
        by_band = paired_distance_band_recall(by_distance_frame(), "Baseline", n_bootstrap=200)

        assert by_band["band"].to_list() == ["0-1", "1-5", "5+"]
        assert by_band["mean_diff"].to_list() == pytest.approx([0.4, 0.2, 0.0])
        assert by_band["n_users"].to_list() == [1, 2, 1]


def make_ranked_per_user() -> pl.DataFrame:
    """A tie-free model at two seeds and a fully tied baseline over ten scored candidates per user."""
    scored_higher = [
        ("MLP", 42, {1: [0, 3], 2: [1]}),
        ("MLP", 43, {1: [1, 4], 2: [0]}),
    ]
    rows = [
        {
            "name": name,
            "stage": PER_USER_STAGE,
            "seed": seed,
            "user_id": user,
            "pos_n_scored_higher": counts,
            "pos_n_tied": [0] * len(counts),
        }
        for name, seed, users in scored_higher
        for user, counts in users.items()
    ]
    rows += [
        {
            "name": "Uniform",
            "stage": PER_USER_STAGE,
            "seed": None,
            "user_id": user,
            "pos_n_scored_higher": [0] * n_pos,
            "pos_n_tied": [9] * n_pos,
        }
        for user, n_pos in [(1, 2), (2, 1)]
    ]

    return pl.DataFrame(rows)


class TestRecallCurve:
    def test_one_row_per_user_and_cutoff(self):
        recalls = recall_at_ks(make_ranked_per_user(), [1, 2, 5])
        assert recalls.height == 6 * 3

    def test_without_ties_a_positive_counts_when_fewer_than_k_score_above(self):
        recalls = recall_at_ks(make_ranked_per_user(), [1, 2, 5]).filter(
            (pl.col("name") == "MLP") & (pl.col("seed") == 42) & (pl.col("user_id") == 1)
        )
        assert recalls["recall_at_k"].to_list() == pytest.approx([0.5, 0.5, 1.0])

    def test_a_fully_tied_model_gets_k_over_the_candidate_count(self):
        recalls = recall_at_ks(make_ranked_per_user(), [1, 2, 20]).filter(
            (pl.col("name") == "Uniform") & (pl.col("user_id") == 1)
        )
        assert recalls["recall_at_k"].to_list() == pytest.approx([0.1, 0.2, 1.0])

    def test_a_partial_tie_spreads_the_positive_over_its_tied_positions(self):
        """Two candidates above and three tied: the positive sits at position 3, 4, 5 or 6 with equal chance."""
        per_user = pl.DataFrame({
            "name": ["MLP"],
            "stage": [PER_USER_STAGE],
            "seed": [42],
            "user_id": [1],
            "pos_n_scored_higher": [[2]],
            "pos_n_tied": [[3]],
        })
        recalls = recall_at_ks(per_user, [2, 3, 4, 6])
        assert recalls["recall_at_k"].to_list() == pytest.approx([0.0, 0.25, 0.5, 1.0])

    def test_curve_reports_mean_and_sd_over_seeds(self):
        row = recall_curve(make_ranked_per_user(), [1]).filter(pl.col("name") == "MLP").to_dicts()[0]

        assert row["n_seeds"] == 2
        assert row["mean"] == pytest.approx(0.375)
        assert row["sd"] == pytest.approx(pl.Series([0.25, 0.5]).std())

    def test_curve_is_ordered_by_cutoff(self):
        curve = recall_curve(make_ranked_per_user(), [5, 1, 2]).filter(pl.col("name") == "MLP")
        assert curve["k"].to_list() == [1, 2, 5]

    def test_paired_comparison_at_each_cutoff(self):
        paired = paired_comparison_by_k(make_ranked_per_user(), "Uniform", [1, 2], n_bootstrap=200)

        assert paired.columns[0] == "k"
        assert dict(zip(paired["k"], paired["mean_diff"], strict=True)) == pytest.approx({1: 0.275, 2: 0.55})


class TestWithRankingMetrics:
    def test_derives_the_per_user_mean_of_the_positive_contributions(self):
        """User 1 at seed 42 has positives at ranks 1 and 4: (50/50 + 47/50) / 2."""
        per_user = with_ranking_metrics(make_ranked_per_user(), BUDGET)
        row = per_user.filter((pl.col("name") == "MLP") & (pl.col("seed") == 42) & (pl.col("user_id") == 1))

        assert row["avg_recall"].to_list() == pytest.approx([(1.0 + 0.94) / 2])
        assert row["pos_avg_recall"].to_list() == [pytest.approx([1.0, 0.94])]

    def test_keeps_the_rows_in_order(self):
        per_user = make_ranked_per_user()

        derived = with_ranking_metrics(per_user, BUDGET)

        assert derived.drop("avg_recall", "pos_avg_recall").equals(per_user)

    def test_a_fully_tied_user_gets_the_mean_over_its_tied_ranks(self):
        """Ten tied candidates and K = 50: the positive is surely in the top k for k >= 10."""
        per_user = with_ranking_metrics(make_ranked_per_user(), BUDGET).filter(pl.col("name") == "Uniform")

        expected = (sum(k / 10 for k in range(1, 10)) + 41) / 50
        assert per_user["avg_recall"].to_list() == pytest.approx([expected, expected])


def synthetic_test_run(budget: int, num_graphs: int = 6, num_nodes: int = 40) -> tuple[dict, pl.DataFrame]:
    """Test an untrained module on graphs with coarse scores, returning its logged metrics and per-user rows."""
    generator = torch.Generator().manual_seed(0)
    graphs = []
    for user in range(num_graphs):
        x = torch.rand(num_nodes, 2, generator=generator)
        x[:, 0] = 0.0
        x[0, 0] = 1.0
        y = (torch.rand(num_nodes, 1, generator=generator) < 0.2).float()
        y[0] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=y)
        graph.graph_x = torch.zeros(1, 1)
        graph.user_id = torch.tensor([user])
        graphs.append(graph)

    idx = torch.arange(num_nodes)
    module = ActivityGraphModule(
        model=NodeMLP(num_layers=2, in_channels=3, hidden_channels=8, out_channels=1),
        lr=1e-3,
        is_home_idx=0,
        home_hop_distance=(idx[:, None] - idx[None, :]).abs().float(),
        max_recall_k=budget,
        recall_ks=[1, 5],
    )
    coarse_logits = (torch.rand(num_graphs * num_nodes, 1, generator=generator) * 4).round()
    module.compute_logits = lambda _batch: coarse_logits

    logged: dict = {}
    module.log = lambda key, val, **kw: logged.__setitem__(key, val)
    module.log_dict = lambda mapping, **kw: logged.update(mapping)

    with torch.no_grad():
        module.test_step(pyg.data.Batch.from_data_list(graphs), 0)
    module.on_test_epoch_end()

    per_user = pl.DataFrame(module.per_user_columns).with_columns(stage=pl.lit(PER_USER_STAGE))
    return {key: float(value) for key, value in logged.items()}, per_user


class TestPostHocMatchesLogged:
    def test_average_recall(self):
        """The coarse scores tie, so this also checks that both paths resolve ties the same way."""
        logged, per_user = synthetic_test_run(budget=10)

        derived = with_ranking_metrics(per_user, 10)

        assert (per_user["pos_n_tied"].list.max() > 0).any()
        assert derived["avg_recall"].mean() == pytest.approx(logged["test_avg_recall@10"], abs=1e-6)

    def test_recall_at_k(self):
        logged, per_user = synthetic_test_run(budget=10)

        recalls = recall_at_ks(per_user.with_columns(name=pl.lit("MLP"), seed=pl.lit(42)), [1, 5])

        for k in (1, 5):
            post_hoc = recalls.filter(pl.col("k") == k)["recall_at_k"].mean()
            assert post_hoc == pytest.approx(logged[f"test_recall@{k}"], abs=1e-6)


class TestAppendBaselines:
    def _extra_run(self) -> pl.DataFrame:
        """A later run holding a new baseline, a seeded baseline and a learned model trained there."""
        return pl.DataFrame({
            "name": ["Gravity", "Gravity", "HomeZoneMF", "HomeZoneMF", "MLP-new", "MLP-new"],
            "stage": [PER_USER_STAGE] * 6,
            "user_id": [1, 2, 1, 2, 1, 2],
            "seed": [None, None, 42, 42, 42, 42],
            "is_baseline": [True, True, True, True, False, False],
            "r_precision": [0.25, 0.75, 0.5, 0.5, 1.0, 1.0],
        })

    def test_appends_only_the_baseline_rows(self):
        combined = append_baselines(make_per_user(), self._extra_run())

        assert combined.filter(pl.col("name") == "Gravity").height == 2
        assert combined.filter(pl.col("name") == "HomeZoneMF").height == 2
        assert combined.filter(pl.col("name") == "MLP-new").is_empty()

    def test_the_appended_baseline_pairs_by_user(self):
        combined = append_baselines(make_per_user(), self._extra_run())

        compared = paired_comparison(combined, "Gravity", PER_USER_METRIC, n_bootstrap=200)

        assert compared.filter(pl.col("name") == "MLP")["n_users"].item() == 2

    def test_a_name_in_both_runs_raises(self):
        extra = self._extra_run().with_columns(name=pl.lit("Baseline"))

        with pytest.raises(ValueError, match="Baseline"):
            append_baselines(make_per_user(), extra)


class TestLoadReportRun:
    def _write_runs(self, path) -> None:
        per_user = make_ranked_per_user().filter(pl.col("name") == "MLP")
        extra = make_ranked_per_user().filter(pl.col("name") == "Uniform")
        aggregate = pl.DataFrame({"name": ["MLP"], "stage": ["test"], "seed": [42], "test_r_precision": [0.5]})
        extra_aggregate = pl.DataFrame(
            {"name": ["Uniform"], "stage": ["test"], "seed": [None], "test_r_precision": [0.1]},
            schema_overrides={"seed": pl.Int64},
        )
        for run, aggregate_frame, per_user_frame in [(1, aggregate, per_user), (2, extra_aggregate, extra)]:
            aggregate_frame.write_parquet(path / f"GenevaTPG-results-{run}.parquet")
            per_user_frame.write_parquet(path / f"GenevaTPG-per-user-{run}.parquet")

        make_score_vectors([[0.1, 0.9]]).write_parquet(path / "GenevaTPG-scores-1.parquet")

    def test_extra_runs_without_score_vectors_are_skipped(self, tmp_path):
        self._write_runs(tmp_path)

        path = Path(tmp_path)
        scores1 = load_score_vectors(path, "GenevaTPG", 1)
        scores = scores1

        assert scores["name"].unique().to_list() == ["MLP-dist"]


class TestLiftOverReference:
    def test_divides_by_the_reference_mean_at_the_same_k(self):
        lifted = lift_over_reference(make_ranked_per_user(), "Uniform", [1, 2])
        mlp = lifted.filter(pl.col("name") == "MLP")

        assert dict(zip(mlp["k"], mlp["lift"], strict=True)) == pytest.approx({1: 0.375 / 0.1, 2: 0.75 / 0.2})
        assert lifted.filter(pl.col("name") == "Uniform")["lift"].to_list() == pytest.approx([1.0, 1.0])

    def test_missing_reference_raises(self):
        with pytest.raises(KeyError):
            lift_over_reference(make_ranked_per_user(), "NotAModel", [1])


class TestHomeZones:
    def test_counts_users_by_argmax_of_the_home_column(self):
        # Four users over three nodes; is_home is column 1. Homes: node 2, 2, 0, 2.
        row = home_zone_summary(make_dataset([2, 2, 0, 2])).to_dicts()[0]
        assert row["n_users"] == 4
        assert row["n_occupied_zones"] == 2
        assert row["max"] == 3

    def test_summary_divides_by_occupied_zones_not_all_nodes(self):
        dataset = make_dataset([0] * 30 + [1] * 10, n_nodes=100)
        row = home_zone_summary(dataset).to_dicts()[0]
        assert row["n_users"] == 40
        assert row["n_occupied_zones"] == 2
        assert row["users_per_occupied_zone"] == 20.0
        assert row["users_per_node"] == 0.4

    def test_occupancy_shares_are_over_users(self):
        dataset = make_dataset([0] * 60 + [1] * 20 + [2] * 5)
        row = home_zone_summary(dataset).to_dicts()[0]
        assert row["share_users_ge_10"] == 80 / 85
        assert row["share_users_ge_60"] == 60 / 85

    def test_thresholds_are_configurable(self):
        dataset = make_dataset([0] * 60 + [1] * 20 + [2] * 5)
        row = home_zone_summary(dataset, occupancy_thresholds=[5]).to_dicts()[0]
        assert set(row) & {"share_users_ge_5"} == {"share_users_ge_5"}
        assert row["share_users_ge_5"] == 1.0
