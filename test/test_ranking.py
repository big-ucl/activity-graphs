"""Unit tests for the shared tie-aware ranking statistics."""

import numpy as np
import pytest
import torch

from activitygraphs.ml.ranking import average_recall, expected_recall_at_k, positive_rank_stats

BUDGET = 50


def mean_recall_over_ks(n_higher: np.ndarray, n_tied: np.ndarray, budget: int) -> np.ndarray:
    """Reference: mean of the tie-aware recall@k over k = 1..budget, one entry per positive."""
    ks = np.arange(1, budget + 1)
    return expected_recall_at_k(n_higher[:, None], n_tied[:, None], ks[None, :]).mean(axis=1)


def per_user_average_recall(scores: torch.Tensor, target: torch.Tensor, budget: int = BUDGET) -> float:
    n_higher, n_tied = positive_rank_stats(scores, target)
    return float(average_recall(n_higher, n_tied, budget).mean())


class TestPositiveRankStats:
    def test_counts_higher_and_tied_candidates(self):
        n_higher, n_tied = positive_rank_stats(torch.tensor([0.9, 0.5, 0.5, 0.5, 0.1]), torch.tensor([0, 1, 0, 1, 0]))

        assert n_higher.tolist() == [1, 1]
        assert n_tied.tolist() == [2, 2]

    def test_invalid_entries_are_not_candidates(self):
        scores = torch.tensor([0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        valid = torch.tensor([False, True, True, True])

        n_higher, n_tied = positive_rank_stats(scores, target, valid)

        assert n_higher.tolist() == [1]
        assert n_tied.tolist() == [0]

    def test_batched_rows_match_one_user_at_a_time(self):
        generator = torch.Generator().manual_seed(0)
        scores = torch.randint(0, 5, (4, 12), generator=generator).float()
        target = torch.rand(4, 12, generator=generator) < 0.3
        valid = torch.rand(4, 12, generator=generator) < 0.8

        n_higher, n_tied = positive_rank_stats(scores, target, valid)

        expected = [positive_rank_stats(scores[u][valid[u]], target[u][valid[u]]) for u in range(4)]
        assert n_higher.tolist() == torch.cat([h for h, _ in expected]).tolist()
        assert n_tied.tolist() == torch.cat([t for _, t in expected]).tolist()


class TestAverageRecall:
    def test_a_positive_at_rank_one_contributes_one(self):
        scores = torch.linspace(1.0, 0.0, 100)
        target = torch.zeros(100, dtype=torch.long)
        target[0] = 1

        assert per_user_average_recall(scores, target) == pytest.approx(1.0)

    def test_a_positive_at_rank_budget_contributes_one_over_budget(self):
        scores = torch.linspace(1.0, 0.0, 100)
        target = torch.zeros(100, dtype=torch.long)
        target[BUDGET - 1] = 1

        assert per_user_average_recall(scores, target) == pytest.approx(1 / BUDGET)

    def test_a_positive_past_the_budget_contributes_nothing(self):
        scores = torch.linspace(1.0, 0.0, 100)
        target = torch.zeros(100, dtype=torch.long)
        target[BUDGET] = 1

        assert per_user_average_recall(scores, target) == pytest.approx(0.0)

    def test_two_positives_average_their_contributions(self):
        """Ranks 1 and 11 contribute 50/50 and 40/50."""
        scores = torch.linspace(1.0, 0.0, 100)
        target = torch.zeros(100, dtype=torch.long)
        target[0] = 1
        target[10] = 1

        assert per_user_average_recall(scores, target) == pytest.approx((1.0 + 0.8) / 2)

    def test_matches_the_closed_form_without_ties(self):
        """``(K + 1 - r) / K`` for a positive at rank ``r <= K``."""
        n_higher = np.arange(0, 60)

        result = average_recall(n_higher, np.zeros_like(n_higher), BUDGET)

        np.testing.assert_allclose(result, np.maximum(0, BUDGET - n_higher) / BUDGET)

    @pytest.mark.parametrize("n_higher", [0, 3, 45, 48, 49, 50, 70])
    @pytest.mark.parametrize("n_tied", [0, 1, 4, 9, 200])
    def test_ties_equal_the_mean_of_the_tie_aware_recall_over_k(self, n_higher, n_tied):
        h, t = np.array([n_higher]), np.array([n_tied])

        np.testing.assert_allclose(average_recall(h, t, BUDGET), mean_recall_over_ks(h, t, BUDGET))

    def test_torch_and_numpy_agree(self):
        rng = np.random.default_rng(0)
        n_higher = rng.integers(0, 80, size=200)
        n_tied = rng.integers(0, 30, size=200)

        from_numpy = average_recall(n_higher, n_tied, BUDGET)
        from_torch = average_recall(torch.as_tensor(n_higher), torch.as_tensor(n_tied), BUDGET)

        np.testing.assert_allclose(from_torch.numpy(), from_numpy)
        np.testing.assert_allclose(
            expected_recall_at_k(torch.as_tensor(n_higher), torch.as_tensor(n_tied), 5).numpy(),
            expected_recall_at_k(n_higher, n_tied, 5),
        )
