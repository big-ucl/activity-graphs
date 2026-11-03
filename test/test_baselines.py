"""Tests for frequency-based baseline models."""

import pytest
import torch
from graphs import expand_users, make_locations, make_users

from activitygraphs.datasets import Users
from activitygraphs.ml.baselines import ConditionalVisitFrequencyBaseline, VisitFrequencyBaseline
from activitygraphs.ml.baselines.common import fit_smoothing_strength, smoothed_rate
from activitygraphs.ml.popularity import home_excluded_visit_rate, home_visit_counts

NUM_LOCATION_FEATURES = 2
IS_HOME_IDX = 0


def random_users(num_users: int, num_nodes: int, seed: int = 0) -> Users:
    """Return users with one random home node each and random visit labels."""
    rng = torch.Generator().manual_seed(seed)
    home_node = torch.randint(0, num_nodes, (num_users,), generator=rng)
    visited = torch.randint(0, 2, (num_users, num_nodes), generator=rng).float()
    return make_users(visited, home_node)


def expand(users: Users):
    """Return one copy of a graph of ``users``' node count per user, with random location features and no edges."""
    num_nodes = users.visited.shape[1]
    x = torch.rand(num_nodes, NUM_LOCATION_FEATURES, generator=torch.Generator().manual_seed(1))
    locations = make_locations(x, torch.zeros(2, 0, dtype=torch.long))
    return expand_users(locations, users)


@pytest.fixture
def batch_uniform():
    """One copy of a 5-node graph for each of 4 users."""
    return expand(random_users(num_users=4, num_nodes=5))


class TestVisitFrequencyBaseline:
    def test_forward_shape(self, batch_uniform):
        num_nodes = 5
        baseline = VisitFrequencyBaseline(num_nodes=num_nodes)
        baseline.fit(random_users(8, num_nodes))
        out = baseline(batch_uniform)
        assert out.shape == (batch_uniform.num_nodes, 1)

    def test_forward_values_in_logit_range(self, batch_uniform):
        num_nodes = 5
        baseline = VisitFrequencyBaseline(num_nodes=num_nodes)
        baseline.fit(random_users(8, num_nodes))
        out = baseline(batch_uniform)
        assert torch.isfinite(out).all()

    def test_scores_are_the_home_excluded_visit_rate(self, batch_uniform):
        users = random_users(8, 5)
        baseline = VisitFrequencyBaseline(num_nodes=5)
        baseline.fit(users)

        expected = torch.logit(home_excluded_visit_rate(users.visited, users.home_node))
        out = baseline(batch_uniform)

        torch.testing.assert_close(out.squeeze(-1), expected.repeat(batch_uniform.num_graphs))


class TestConditionalVisitFrequencyBaseline:
    def test_forward_shape(self):
        num_nodes = 5
        users = random_users(num_users=8, num_nodes=num_nodes)
        batched = expand(users.subset(list(range(4))))

        baseline = ConditionalVisitFrequencyBaseline(num_nodes=num_nodes, is_home_idx_user_node=IS_HOME_IDX)
        baseline.fit(users, users)
        out = baseline(batched)
        assert out.shape == (batched.num_nodes, 1)


def split_of(labels: list[list[int]], homes: list[int]) -> Users:
    return make_users(torch.tensor(labels).float(), torch.tensor(homes))


class TestSmoothing:
    COUNTS = torch.tensor([[0.0, 3.0, 1.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    USERS = torch.tensor([4.0, 0.0, 2.0])
    PRIOR = torch.tensor([0.3, 0.2, 0.1])

    def test_counts_are_indexed_by_home_then_node(self):
        counts, users = home_visit_counts(torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), torch.tensor([0, 0, 2]))

        assert counts.tolist() == [[2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        assert users.tolist() == [2.0, 0.0, 1.0]

    def test_a_vanishing_strength_gives_the_per_home_frequency(self):
        rate = smoothed_rate(self.COUNTS, self.USERS, self.PRIOR, 1e-9)

        assert rate[0].tolist() == pytest.approx([1e-6, 0.75, 0.25])
        assert rate[2].tolist() == pytest.approx([1.0 - 1e-6, 1e-6, 1e-6])

    def test_a_large_strength_gives_the_prior(self):
        rate = smoothed_rate(self.COUNTS, self.USERS, self.PRIOR, 1e9)

        torch.testing.assert_close(rate, self.PRIOR.expand(3, 3))

    def test_an_unseen_home_gets_the_prior_exactly(self):
        rate = smoothed_rate(self.COUNTS, self.USERS, self.PRIOR, 0.5)

        torch.testing.assert_close(rate[1], self.PRIOR)

    def test_strength_is_small_when_validation_follows_the_home(self):
        """Every resident of home 0 visits node 1 and no one else does, in training and validation alike."""
        train = split_of([[1, 1, 0]] * 5 + [[0, 0, 1]] * 5, [0] * 5 + [2] * 5)
        val = split_of([[1, 1, 0]] * 3 + [[0, 0, 1]] * 3, [0] * 3 + [2] * 3)
        counts, users = home_visit_counts(train.visited, train.home_node)

        strength = fit_smoothing_strength(counts, users, val, torch.full((6, 3), 0.5))

        assert strength == pytest.approx(0.01)

    def test_strength_is_large_when_validation_follows_the_prior(self):
        """Training residents of home 0 look home-specific, but validation residents visit at the prior rate."""
        train = split_of([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], [0, 0, 0, 0])
        val = split_of([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], [0, 0, 0, 0])
        counts, users = home_visit_counts(train.visited, train.home_node)

        strength = fit_smoothing_strength(counts, users, val, torch.full((4, 3), 0.5))

        assert strength == pytest.approx(1000.0)


class TestConditionalVisitFrequencySmoothing:
    def test_an_unseen_home_is_scored_with_the_visit_frequency(self):
        train = split_of([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], [0, 0, 1])
        val = split_of([[1, 1, 0, 1], [0, 1, 0, 0]], [0, 1])
        conditional = ConditionalVisitFrequencyBaseline(num_nodes=4, is_home_idx_user_node=IS_HOME_IDX).fit(train, val)
        frequency = VisitFrequencyBaseline(num_nodes=4).fit(train)

        unseen = expand(split_of([[0, 0, 0, 1]], [3]))

        torch.testing.assert_close(conditional(unseen), frequency(unseen))

    def test_the_strength_is_fitted(self):
        train = split_of([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], [0, 0, 1])

        conditional = ConditionalVisitFrequencyBaseline(num_nodes=4, is_home_idx_user_node=IS_HOME_IDX).fit(
            train, train
        )

        assert conditional.strength is not None
        assert conditional.logits.shape == (4, 4)
