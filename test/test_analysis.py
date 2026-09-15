"""Tests for the post-hoc analysis of experiment outputs (seed spread, pairing, hop bands, home zones)."""

import math
from dataclasses import dataclass

import numpy as np

import polars as pl
import pytest
import torch

from activitygraphs.analysis import (
    DROPPED_USERS_COLUMN,
    REALISED_SIZE_COLUMN,
    SCORED_USERS_COLUMN,
    _average_metric_over_seeds,
    _diff_sd_per_seed,
    _extract_band_and_metric,
    _latest_run,
    aggregate_metrics,
    band_contributions,
    band_decomposition,
    compute_hop_band_table,
    paired_comparison_by_band,
    paired_comparison_by_k,
    recall_at_ks,
    recall_curve,
    home_zone_summary,
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
from activitygraphs.ml.training import PER_USER_STAGE, SCORE_VECTOR_STAGE

MAIN_METRIC = "test_r_precision"
PER_USER_METRIC = "r_precision"


def make_aggregate() -> pl.DataFrame:
    """Two seeds of one model plus a deterministic baseline, with two hop bands populated."""
    rows = [
        {"name": "MLP", "stage": "test", "seed": 42, "test_r_precision": 0.40},
        {"name": "MLP", "stage": "test", "seed": 43, "test_r_precision": 0.50},
        {"name": "Baseline", "stage": "test", "seed": None, "test_r_precision": 0.30},
        {"name": "MLP", "stage": "fit", "seed": 42, "test_r_precision": None},
    ]
    bands = {
        "test_hop_0-2_recall@2": [0.9, 1.0, 0.5, None],
        "test_hop_0-2_n_pos": [30.0, 30.0, 30.0, None],
        "test_hop_13+_recall@2": [0.1, 0.2, 0.05, None],
        "test_hop_13+_n_pos": [10.0, 10.0, 10.0, None],
    }

    return pl.DataFrame(rows).with_columns(
        **{col: pl.Series(values) for col, values in bands.items()},
        **{SCORED_USERS_COLUMN: pl.lit(90.0), DROPPED_USERS_COLUMN: pl.lit(10.0)},
    )


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
            {"name": "MLP-dist", "stage": "fit", "epoch": 0, "train_r_precision": 0.3},
            {"name": "MLP-dist", "stage": "fit", "epoch": 1, "train_r_precision": 0.99},
            {"name": "MLP-dist", "stage": "fit", "epoch": 2, "train_r_precision": 0.97},
            {"name": "GATSkip", "stage": "fit", "epoch": 0, "train_r_precision": 0.2},
            {"name": "GATSkip", "stage": "fit", "epoch": 1, "train_r_precision": 0.25},
        ])

    def test_reports_best_and_final_per_model(self):
        health = self._fit_rows().pipe(check_overfit_health)

        assert health["name"].to_list() == ["GATSkip", "MLP-dist"]  # worst first
        assert health["best_train_r_precision"].to_list() == pytest.approx([0.25, 0.99])
        assert health["final_train_r_precision"].to_list() == pytest.approx([0.25, 0.97])

    def test_raises_without_a_train_ranking_column(self):
        with pytest.raises(KeyError, match="log_train_ranking"):
            check_overfit_health(make_aggregate())


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
            rows.append({"name": "NodeMarginal", "seed": None, "user_id": user_id, "scores": list(self.POPULARITY)})

        return pl.DataFrame(rows, schema_overrides={"seed": pl.Int64})

    def _per_user(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": list(self.HOMES),
            "home_node": list(self.HOMES.values()),
            "stage": [PER_USER_STAGE] * len(self.HOMES),
        })

    def test_separates_a_collapsed_model_from_a_home_aware_one(self):
        health = check_model_health(self._score_vectors(), self._per_user())
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
        health = check_model_health(self._score_vectors(), self._per_user())
        by_name = {row["name"]: row for row in health.to_dicts()}

        assert by_name["HomeAware"]["n_seeds"] == 2
        assert by_name["NodeMarginal"]["n_seeds"] == 1  # deterministic, no seed
        assert by_name["HomeAware"]["home_is_top1_sd"] == pytest.approx(0.0)

    def test_the_collapsed_models_sort_above_the_home_aware_one(self):
        """`NodeMarginal` is collapsed by construction, so it ties `Collapsed` at the top."""
        health = check_model_health(self._score_vectors(), self._per_user())

        assert health["name"].to_list() == ["Collapsed", "NodeMarginal", "HomeAware"]

    def test_baseline_reproduces_the_popularity_ordering(self):
        """NodeMarginal is the popularity reference, so it must score 1.0 against itself - a control on the check."""
        health = check_model_health(self._score_vectors(), self._per_user())
        by_name = {row["name"]: row for row in health.to_dicts()}

        assert by_name["NodeMarginal"]["popularity_corr"] == pytest.approx(1.0)

    def test_raises_when_the_popularity_reference_is_missing(self):
        scores = self._score_vectors().filter(pl.col("name") != "NodeMarginal")

        with pytest.raises(KeyError, match="NodeMarginal"):
            check_model_health(scores, self._per_user())


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


class TestHopBands:
    def test_splits_metric_names_containing_underscores(self):
        assert _extract_band_and_metric("test_hop_3-5_recall@2") == ("test_hop_3-5_recall@2", "3-5", 3, "recall@2")
        assert _extract_band_and_metric("test_hop_13+_n_pos") == ("test_hop_13+_n_pos", "13+", 13, "n_pos")

    def test_share_of_positives_uses_the_band_counts(self):
        table = compute_hop_band_table(make_aggregate(), models=["MLP"])
        shares = dict(zip(table["band"], table["share_of_pos"], strict=True))
        assert shares["0-2"] == 0.75  # 30 of 40 positives
        assert shares["13+"] == 0.25

    def test_reports_mean_and_sd_within_a_band(self):
        table = compute_hop_band_table(make_aggregate(), models=["MLP"])
        row = table.filter((pl.col("band") == "0-2") & (pl.col("metric") == "recall@2")).to_dicts()[0]
        assert abs(row["mean"] - 0.95) < 1e-9
        assert abs(row["sd"] - 0.0707106) < 1e-6

    def test_n_pos_is_a_column_not_a_metric_row(self):
        table = compute_hop_band_table(make_aggregate())
        assert "n_pos" not in table["metric"].to_list()

    def test_bands_are_ordered_by_their_lower_bound(self):
        table = compute_hop_band_table(make_aggregate(), models=["MLP"])
        assert table["band"].to_list() == ["0-2", "13+"]  # not lexicographic, which puts "13+" first

    def test_models_filter_restricts_the_table(self):
        table = compute_hop_band_table(make_aggregate(), models=["Baseline"])
        assert table["name"].unique().to_list() == ["Baseline"]


def make_positive_per_user() -> pl.DataFrame:
    """Two users scored by a model at two seeds and by a baseline once, with each positive's hop distance and flags."""
    user_hops = {1: [1.0, float("inf")], 2: [3.0, 5.0]}
    top_r_flags = [
        ("MLP", 42, {1: [True, True], 2: [False, False]}),
        ("MLP", 43, {1: [True, False], 2: [True, False]}),
        ("Baseline", None, {1: [True, False], 2: [False, False]}),
    ]
    rows = [
        {
            "name": name,
            "stage": PER_USER_STAGE,
            "seed": seed,
            "user_id": user,
            "n_pos": len(hops),
            "r_precision": sum(flags[user]) / len(hops),
            "recall": 1.0,
            "pos_hops": hops,
            "pos_in_top_r": flags[user],
            "pos_in_top_k": [True] * len(hops),
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

    def test_recall_reads_the_top_k_flag(self):
        table = band_decomposition(make_positive_per_user(), "recall").filter(pl.col("name") == "Baseline")
        assert table["mean"].sum() == pytest.approx(1.0)

    def test_a_metric_without_a_flag_raises(self):
        with pytest.raises(KeyError, match="decomposable metrics"):
            band_contributions(make_positive_per_user(), "ndcg")

    def test_paired_band_differences_sum_to_the_overall_difference(self):
        per_user = make_positive_per_user()
        by_band = paired_comparison_by_band(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200)
        overall = paired_comparison(per_user, "Baseline", PER_USER_METRIC, n_bootstrap=200)["mean_diff"][0]

        diffs = dict(zip(by_band["band"], by_band["mean_diff"], strict=True))
        assert diffs == pytest.approx({"0-2": 0.0, "3-5": 0.125, "13+": 0.125})
        assert sum(diffs.values()) == pytest.approx(overall)
        assert by_band["n_users"].to_list() == [2, 2, 2]


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
