"""Tests for the home-excluded node popularity."""

import pytest
import torch

from activitygraphs.ml.popularity import POPULARITY_EPS, home_excluded_visit_rate, popularity_logit

# Three users over three nodes. Users 0 and 1 live at node 0, user 2 at node 1; every home is labelled visited.
LABELS = torch.tensor([
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])
HOME_IDX = torch.tensor([0, 0, 1])


class TestHomeExcludedVisitRate:
    def test_hand_computed_rates(self):
        """Node 0: only user 2 is a candidate and visits. Node 1: users 0-1 are, one visits. Node 2: all three, two."""
        rate = home_excluded_visit_rate(LABELS, HOME_IDX)

        assert rate.tolist() == pytest.approx([1 - POPULARITY_EPS, 0.5, 2 / 3])

    def test_residents_leave_the_numerator_and_the_denominator(self):
        """Without the exclusion node 1 would read 2/3: user 2's home visit would count."""
        assert home_excluded_visit_rate(LABELS, HOME_IDX)[1].item() == pytest.approx(0.5)

    def test_a_node_everyone_lives_at_gets_the_floor(self):
        labels = torch.tensor([[1, 0], [1, 1]])

        rate = home_excluded_visit_rate(labels, torch.tensor([0, 0]))

        assert rate.tolist() == pytest.approx([POPULARITY_EPS, 0.5])

    def test_logit_is_the_logit_of_the_rate(self):
        torch.testing.assert_close(
            popularity_logit(LABELS, HOME_IDX), torch.logit(home_excluded_visit_rate(LABELS, HOME_IDX))
        )
