"""Tests for the distance-decay baselines."""

import dataclasses
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from graphs import edgeless_locations, join_users, make_locations, make_users

from activitygraphs.datasets import Locations, UserBatch, Users
from activitygraphs.models.baselines.gravity import (
    MIN_DISTANCE_M,
    ConditionalGravityBaseline,
    DistanceDecayBaseline,
    GravityBaseline,
    GravityBinnedBaseline,
    LogsumGravityBaseline,
    OpportunityGravityBaseline,
    TravelTimeGravityBaseline,
    _weighted_median,
)
from activitygraphs.models.baselines.popularity import home_excluded_visit_rate
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema

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


def simulate_opportunities(
    n_users: int, n_nodes: int, mass_weights: tuple[float, float], beta_log: float, seed: int = 0
) -> tuple[Users, torch.Tensor, torch.Tensor]:
    """Draw labels from a gravity model with the jobs and residents of each node as mass.

    Node ``n`` is ``offsets[(n - h) % n_nodes]`` metres from home ``h``, as in ``simulate``.

    Returns:
        ``(users, distance_m, opportunities)``, with ``opportunities`` the ``[n_nodes, 2]`` jobs and residents.
    """
    generator = torch.Generator().manual_seed(seed)
    offsets = 200.0 + 19_800.0 * torch.rand(n_nodes, generator=generator)
    offsets[0] = 0.0
    nodes = torch.arange(n_nodes)
    distance_m = offsets[(nodes[None, :] - nodes[:, None]) % n_nodes]
    opportunities = torch.exp(8.0 * torch.rand(n_nodes, 2, generator=generator))

    homes = torch.randint(0, n_nodes, (n_users,), generator=generator)
    mass = opportunities.log1p() @ torch.tensor(mass_weights)
    distances = distance_m[homes].clamp(min=MIN_DISTANCE_M)
    logits = -2.0 + mass - beta_log * distances.log()
    labels = (torch.rand(n_users, n_nodes, generator=generator) < logits.sigmoid()).float()
    labels[torch.arange(n_users), homes] = 1.0

    return make_users(labels, homes), distance_m, opportunities


def opportunity_locations(distance_m: torch.Tensor, opportunities: torch.Tensor) -> Locations:
    """Return a graph with no edges whose nodes are ``distance_m`` metres apart and hold ``opportunities``."""
    num_nodes = distance_m.shape[0]
    x = torch.zeros(num_nodes, NUM_LOCATION_FEATURES)
    edge_index = torch.zeros(2, 0, dtype=torch.long)

    return make_locations(x, edge_index, distance_m=distance_m, opportunities=opportunities)


class TestOpportunityGravity:
    def test_the_fit_recovers_the_simulated_mass_weights(self):
        mass_weights = (0.5, 0.2)
        users, distance_m, opportunities = simulate_opportunities(
            n_users=3000, n_nodes=300, mass_weights=mass_weights, beta_log=1.0
        )
        locations = opportunity_locations(distance_m, opportunities)

        gravity = OpportunityGravityBaseline().fit(users, users, locations)

        assert gravity.mass_weights.tolist() == pytest.approx(mass_weights, abs=0.05)
        assert float(gravity.beta_log[0]) > 0.5

    def test_reports_the_mass_weight_of_each_opportunity_column(self):
        users, distance_m, opportunities = simulate_opportunities(
            n_users=500, n_nodes=50, mass_weights=(0.5, 0.2), beta_log=1.0
        )
        gravity = OpportunityGravityBaseline().fit(users, users, opportunity_locations(distance_m, opportunities))

        params = gravity.fitted_params()

        assert list(params) == ["alpha", "beta_log", "beta_lin", "d0_m", "mass_jobs", "mass_population"]
        assert [params["mass_jobs"], params["mass_population"]] == gravity.mass_weights.tolist()

    def test_the_fitted_model_scores_the_nodes_of_another_graph(self):
        users, distance_m, opportunities = simulate_opportunities(
            n_users=2000, n_nodes=200, mass_weights=(0.5, 0.2), beta_log=1.0
        )
        gravity = OpportunityGravityBaseline().fit(users, users, opportunity_locations(distance_m, opportunities))

        other_distance_m = torch.full((4, 4), 1000.0).fill_diagonal_(0.0)
        other_opportunities = torch.tensor([[0.0, 0.0], [10.0, 10.0], [1000.0, 10.0], [100.0, 100.0]])
        other_locations = opportunity_locations(other_distance_m, other_opportunities)
        batch = join_users(other_locations, make_users(torch.zeros(1, 4), torch.tensor([0])))

        scores = gravity(other_locations, batch).squeeze(0)

        assert scores.shape == (4,)
        assert scores[1:].argsort(descending=True).tolist() == [1, 2, 0]


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
        assert conditional.num_visits.shape == (10, 10)


TRAVEL_TIME_SCHEMA = TensorSchema(
    "travel_times",
    (ColumnSpec("car_time_min", ColumnKind.COUNT), ColumnSpec("car_reachable", ColumnKind.BINARY)),
)


def simulate_travel_times(n_users: int, n_nodes: int, seed: int = 0) -> tuple[Users, torch.Tensor, torch.Tensor]:
    """Draw labels that decay with the car time alone, beside a distance unrelated to the car time.

    The pairs with a car time above 60 min are unreachable and never visited.

    Returns:
        ``(users, distance_m, travel_times_min)``, with ``travel_times_min`` of ``TRAVEL_TIME_SCHEMA``,
        ``[n_nodes, n_nodes, 2]``.
    """
    generator = torch.Generator().manual_seed(seed)
    distance_m = 500.0 + 20_000.0 * torch.rand(n_nodes, n_nodes, generator=generator)
    car_time_min = 2.0 + 80.0 * torch.rand(n_nodes, n_nodes, generator=generator)
    distance_m.fill_diagonal_(0.0)
    car_time_min.fill_diagonal_(0.0)
    car_reachable = (car_time_min <= 60.0).float()
    travel_times_min = torch.stack([car_time_min.clamp(max=60.0), car_reachable], dim=-1)

    homes = torch.randint(0, n_nodes, (n_users,), generator=generator)
    logits = 1.0 - 1.5 * car_time_min[homes].clamp(min=1.0).log()
    labels = (torch.rand(n_users, n_nodes, generator=generator) < logits.sigmoid()).float()
    labels = labels * car_reachable[homes]
    labels[torch.arange(n_users), homes] = 1.0

    return make_users(labels, homes), distance_m, travel_times_min


class TestTravelTimeGravity:
    def test_the_fit_decays_with_the_time_that_drives_the_visits(self):
        users, distance_m, travel_times_min = simulate_travel_times(n_users=2000, n_nodes=60)
        locations = make_locations(
            torch.zeros(60, 1),
            torch.zeros(2, 0, dtype=torch.long),
            distance_m=distance_m,
            travel_times_min=travel_times_min,
        )

        gravity = TravelTimeGravityBaseline(TRAVEL_TIME_SCHEMA).fit(users, users, locations)
        distance_decay, car_decay = gravity.beta_log.tolist()

        assert car_decay == pytest.approx(1.5, abs=0.3)
        assert abs(distance_decay) < 0.3
        assert gravity.gamma.item() > 2.0

    def test_the_fit_minimises_the_loss_over_every_user_pair(self):
        users, distance_m, travel_times_min = simulate_travel_times(n_users=300, n_nodes=12)
        locations = make_locations(
            torch.zeros(12, 1),
            torch.zeros(2, 0, dtype=torch.long),
            distance_m=distance_m,
            travel_times_min=travel_times_min,
        )
        gravity = TravelTimeGravityBaseline(TRAVEL_TIME_SCHEMA).fit(users, users, locations)
        batch = join_users(locations, users)
        logits = gravity(locations, batch)
        not_home = ~batch.is_home

        def user_pair_loss(intercept_shift: float) -> float:
            shifted_logits = logits[not_home] + intercept_shift
            return float(F.binary_cross_entropy_with_logits(shifted_logits, users.visited[not_home]))

        assert user_pair_loss(0.0) < user_pair_loss(0.05)
        assert user_pair_loss(0.0) < user_pair_loss(-0.05)

    def test_forward_scores_a_shorter_reachable_trip_higher(self):
        users, distance_m, travel_times_min = simulate_travel_times(n_users=2000, n_nodes=60)
        locations = make_locations(
            torch.zeros(60, 1),
            torch.zeros(2, 0, dtype=torch.long),
            distance_m=distance_m,
            travel_times_min=travel_times_min,
        )
        gravity = TravelTimeGravityBaseline(TRAVEL_TIME_SCHEMA).fit(users, users, locations)
        batch = join_users(locations, users)

        logits = gravity(locations, batch)
        home = int(users.home_node[0])
        car_time = travel_times_min[home, :, 0]
        is_reachable = travel_times_min[home, :, 1] > 0.5
        is_reachable[home] = False
        fastest_node_idx = car_time.masked_fill(~is_reachable, float("inf")).argmin()
        slowest_node_idx = car_time.masked_fill(~is_reachable, -1.0).argmax()
        unreachable_node_idx = torch.nonzero(~is_reachable & (torch.arange(60) != home))[0, 0]

        assert logits[0, fastest_node_idx] > logits[0, slowest_node_idx] > logits[0, unreachable_node_idx]


class TestWeightedMedian:
    def test_equals_the_median_of_the_repeated_rows(self):
        values = torch.tensor([[5.0, 1.0], [1.0, 7.0], [3.0, 2.0], [9.0, 4.0]])
        weights = torch.tensor([1.0, 4.0, 2.0, 2.0])
        repeated_values = values.repeat_interleave(weights.long(), dim=0)

        torch.testing.assert_close(_weighted_median(values, weights), repeated_values.median(dim=0).values)


LOGSUM_SCHEMA = TensorSchema(
    "travel_times",
    (
        ColumnSpec("pt_time_min", ColumnKind.COUNT),
        ColumnSpec("car_time_min", ColumnKind.COUNT),
        ColumnSpec("pt_reachable", ColumnKind.BINARY),
        ColumnSpec("car_reachable", ColumnKind.BINARY),
    ),
)


def simulate_shortest_mode(n_users: int, n_nodes: int, seed: int = 0) -> tuple[Users, torch.Tensor]:
    """Draw labels that decay with the shorter of two mode times, both drawn independently of each other.

    Returns:
        ``(users, travel_times_min)``, with ``travel_times_min`` of ``LOGSUM_SCHEMA``, ``[n_nodes, n_nodes, 4]``.
    """
    generator = torch.Generator().manual_seed(seed)
    pt_time_min = 5.0 + 120.0 * torch.rand(n_nodes, n_nodes, generator=generator)
    car_time_min = 5.0 + 120.0 * torch.rand(n_nodes, n_nodes, generator=generator)
    shortest_time_min = torch.minimum(pt_time_min, car_time_min)
    flags = torch.ones(n_nodes, n_nodes, 2)
    travel_times_min = torch.cat([pt_time_min.unsqueeze(-1), car_time_min.unsqueeze(-1), flags], dim=-1)

    homes = torch.randint(0, n_nodes, (n_users,), generator=generator)
    logits = 2.0 - 1.5 * shortest_time_min[homes].log()
    labels = (torch.rand(n_users, n_nodes, generator=generator) < logits.sigmoid()).float()
    labels[torch.arange(n_users), homes] = 1.0

    return make_users(labels, homes), travel_times_min


class TestLogsumGravity:
    def test_the_logsum_time_lies_between_the_shortest_and_the_mean_time(self):
        gravity = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=False)
        travel_times_min = torch.tensor([[10.0, 50.0, 1.0, 1.0], [30.0, 30.0, 1.0, 1.0]])

        for theta in [0.01, 0.1, 1.0]:
            logsum = gravity._logsum_times(travel_times_min, theta)

            assert (logsum >= torch.tensor([10.0, 30.0]) - 1e-4).all()
            assert (logsum <= torch.tensor([30.0, 30.0]) + 1e-4).all()

    def test_a_large_scale_brings_the_logsum_time_near_the_shortest_time(self):
        gravity = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=False)
        travel_times_min = torch.tensor([[10.0, 50.0, 1.0, 1.0]])

        logsum = gravity._logsum_times(travel_times_min, theta=1.0)

        torch.testing.assert_close(logsum, torch.tensor([10.0 - math.log(0.5)]), atol=1e-3, rtol=0)

    def test_the_fit_keeps_a_large_scale_when_visits_follow_the_shortest_time(self):
        users, travel_times_min = simulate_shortest_mode(n_users=2000, n_nodes=60)
        locations = make_locations(
            torch.zeros(60, 1), torch.zeros(2, 0, dtype=torch.long), travel_times_min=travel_times_min
        )

        gravity = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=False).fit(users, users, locations)

        assert gravity.theta >= 0.3
        assert gravity.beta_log.item() > 0.5

    def test_forward_scores_a_shorter_trip_by_the_faster_mode_higher(self):
        users, travel_times_min = simulate_shortest_mode(n_users=2000, n_nodes=60)
        locations = make_locations(
            torch.zeros(60, 1), torch.zeros(2, 0, dtype=torch.long), travel_times_min=travel_times_min
        )
        gravity = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=False).fit(users, users, locations)
        batch = join_users(locations, users)

        logits = gravity(locations, batch)
        home = int(users.home_node[0])
        shortest_time = travel_times_min[home, :, :2].min(dim=-1).values
        shortest_time[home] = float("inf")
        fastest_node_idx = shortest_time.argmin()
        shortest_time[home] = -1.0
        slowest_node_idx = shortest_time.argmax()

        assert logits[0, fastest_node_idx] > logits[0, slowest_node_idx]

    def test_the_distance_term_adds_one_scale_and_one_pair_of_decays(self):
        users, travel_times_min = simulate_shortest_mode(n_users=500, n_nodes=30)
        distance_m = 1000.0 * travel_times_min[..., :2].min(dim=-1).values
        locations = make_locations(
            torch.zeros(30, 1),
            torch.zeros(2, 0, dtype=torch.long),
            distance_m=distance_m,
            travel_times_min=travel_times_min,
        )

        gravity = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=True).fit(users, users, locations)
        batch = join_users(locations, users)

        assert gravity.scales.shape == (2,)
        assert gravity.beta_log.shape == (2,)
        assert gravity(locations, batch).shape == (users.num_users, 30)

    def test_the_distance_term_keeps_the_scale_chosen_on_the_logsum_time_alone(self):
        users, travel_times_min = simulate_shortest_mode(n_users=500, n_nodes=30)
        distance_m = 1000.0 * travel_times_min[..., :2].mean(dim=-1)
        locations = make_locations(
            torch.zeros(30, 1),
            torch.zeros(2, 0, dtype=torch.long),
            distance_m=distance_m,
            travel_times_min=travel_times_min,
        )

        without_distance = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=False).fit(users, users, locations)
        with_distance = LogsumGravityBaseline(LOGSUM_SCHEMA, with_distance=True).fit(users, users, locations)

        assert with_distance.theta == without_distance.theta
