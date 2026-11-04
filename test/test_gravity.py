"""Tests for the distance-decay baselines."""

import dataclasses

import numpy as np
import pytest
import torch
from graphs import edgeless_locations, join_users, make_locations, make_users

from activitygraphs.datasets import Locations, UserBatch, Users
from activitygraphs.models.baselines.gravity import (
    MIN_DISTANCE_M,
    ConditionalGravityBaseline,
    DistanceDecayBaseline,
    GravityBaseline,
    GravityBinnedBaseline,
)
from activitygraphs.models.baselines.popularity import home_excluded_visit_rate

NUM_LOCATION_FEATURES = 1


def join(users: Users, distance_m: torch.Tensor) -> tuple[Locations, UserBatch]:
    """Return a graph with no edges whose nodes are ``distance_m`` metres apart, and ``users`` joined to it."""
    num_nodes = distance_m.shape[0]
    x = torch.zeros(num_nodes, NUM_LOCATION_FEATURES)
    locations = make_locations(x, torch.zeros(2, 0, dtype=torch.long), distance_m=distance_m)
    return locations, join_users(locations, users)


def simulate(
    n_users: int, n_nodes: int, beta_log: float, rate_per_m: float, seed: int = 0
) -> tuple[Users, torch.Tensor]:
    """Draw labels from a gravity model with equal popularity everywhere; each user's home is at distance 0.

    Node ``n`` is ``offsets[(n - h) % n_nodes]`` metres from home ``h``, so every node lies at the same set of distances
    from the homes.

    Returns:
        ``(users, distance_m)``, with ``distance_m`` the ``[n_nodes, n_nodes]`` distances in metres.
    """
    generator = torch.Generator().manual_seed(seed)
    offsets = 200.0 + 19_800.0 * torch.rand(n_nodes, generator=generator)
    offsets[0] = 0.0
    nodes = torch.arange(n_nodes)
    distance_m = offsets[(nodes[None, :] - nodes[:, None]) % n_nodes]

    homes = torch.randint(0, n_nodes, (n_users,), generator=generator)
    distances = distance_m[homes]
    logits = 2.0 - beta_log * distances.clamp(min=MIN_DISTANCE_M).log() - rate_per_m * distances
    labels = (torch.rand(n_users, n_nodes, generator=generator) < logits.sigmoid()).float()
    labels[torch.arange(n_users), homes] = 1.0

    return make_users(labels, homes), distance_m


class TestDistanceDecay:
    def test_ranks_nearest_first(self):
        distance_m = torch.zeros(5, 5)
        distance_m[0] = torch.tensor([0.0, 900.0, 300.0, 5000.0, 1200.0])
        locations, batch = join(make_users(torch.zeros(1, 5), torch.tensor([0])), distance_m)

        scores = DistanceDecayBaseline()(locations, batch).squeeze(0)

        assert scores.argsort(descending=True).tolist() == [0, 2, 1, 4, 3]
        assert scores.sigmoid().unique().numel() == 5


class TestGravity:
    def test_the_tanner_fit_recovers_the_simulated_decay(self):
        """``beta_log`` is scale-free; ``beta_lin`` is the per-metre rate expressed in units of ``d0``."""
        beta_log, rate_per_m = 0.8, 1e-4
        users, distance_m = simulate(n_users=3000, n_nodes=400, beta_log=beta_log, rate_per_m=rate_per_m)

        locations = edgeless_locations(len(distance_m), distance_m)
        gravity = GravityBaseline().fit(users, users, locations)

        assert gravity.beta_log == pytest.approx(beta_log, abs=0.1)
        assert gravity.beta_lin == pytest.approx(rate_per_m * gravity.d0, abs=0.1)

    def test_d0_is_the_median_visited_non_home_distance(self):
        users, distance_m = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=0.0)

        locations = edgeless_locations(len(distance_m), distance_m)
        gravity = GravityBaseline(max_iter=1).fit(users, users, locations)

        labels = users.visited
        not_home = torch.ones_like(labels, dtype=torch.bool)
        not_home[torch.arange(len(labels)), users.home_node] = False
        visited = distance_m[users.home_node][not_home & (labels > 0)]
        assert gravity.d0 == pytest.approx(float(visited.median()), rel=1e-3)

    def test_forward_is_the_pair_logit(self):
        users, distance_m = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        locations = edgeless_locations(len(distance_m), distance_m)
        gravity = GravityBaseline().fit(users, users, locations)

        locations, batch = join(users.subset([0, 1]), distance_m)
        out = gravity(locations, batch)[0]

        node = 3
        relative = max(float(batch.home_distance_m[0, node]), MIN_DISTANCE_M) / gravity.d0
        expected = (
            gravity.alpha
            + float(gravity.popularity_logit[node])
            - gravity.beta_log * np.log(relative)
            - gravity.beta_lin * relative
        )
        assert float(out[node]) == pytest.approx(expected, rel=1e-4)


class TestGravityBinned:
    def test_deterrence_is_observed_over_expected_per_bin(self):
        users, distance_m = simulate(n_users=300, n_nodes=12, beta_log=1.0, rate_per_m=0.0)

        locations = edgeless_locations(len(distance_m), distance_m)
        binned = GravityBinnedBaseline(n_bins=4).fit(users, users, locations)

        labels = users.visited
        popularity = home_excluded_visit_rate(labels, users.home_node)
        pair_distances = distance_m[users.home_node].clamp(min=MIN_DISTANCE_M)
        observed, expected = np.zeros(4), np.zeros(4)
        for user in range(len(labels)):
            for node in range(labels.shape[1]):
                if node == int(users.home_node[user]):
                    continue
                b = int((pair_distances[user, node] > binned.edges).sum())
                observed[b] += float(labels[user, node])
                expected[b] += float(popularity[node])

        np.testing.assert_allclose(binned.log_deterrence.numpy(), np.log((observed + 0.5) / expected), rtol=1e-4)

    def test_one_finite_deterrence_per_bin(self):
        users, distance_m = simulate(n_users=300, n_nodes=12, beta_log=1.0, rate_per_m=0.0)

        locations = edgeless_locations(len(distance_m), distance_m)
        binned = GravityBinnedBaseline(n_bins=4).fit(users, users, locations)

        assert binned.edges.shape == (3,)
        assert binned.log_deterrence.shape == (4,)
        assert torch.all(binned.log_deterrence.isfinite())


class TestConditionalGravity:
    def test_an_unseen_home_is_scored_with_the_gravity_prior(self):
        users, distance_m = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        users = dataclasses.replace(users, home_node=users.home_node % 9)
        train, val = users.subset(list(range(150))), users.subset(list(range(150, 200)))
        locations = edgeless_locations(len(distance_m), distance_m)
        gravity = GravityBaseline().fit(users, users, locations)
        conditional = ConditionalGravityBaseline(gravity).fit(train, val, locations)

        unseen = join(make_users(torch.zeros(1, 10), torch.tensor([9])), distance_m)

        prior = gravity(*unseen)
        out = conditional(*unseen)

        non_home = slice(0, 9)
        torch.testing.assert_close(out[:, non_home], prior[:, non_home], rtol=1e-4, atol=1e-4)

    def test_the_strength_is_fitted(self):
        users, distance_m = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        train, val = users.subset(list(range(150))), users.subset(list(range(150, 200)))
        locations = edgeless_locations(len(distance_m), distance_m)
        gravity = GravityBaseline().fit(users, users, locations)

        conditional = ConditionalGravityBaseline(gravity).fit(train, val, locations)

        assert conditional.strength is not None
        assert conditional.counts.shape == (10, 10)
