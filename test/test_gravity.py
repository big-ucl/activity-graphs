"""Tests for the distance-decay baselines."""

import numpy as np
import pytest
import torch
import torch_geometric as pyg

from activitygraphs.ml.baselines.common import extract_per_user_tensors
from activitygraphs.ml.baselines.gravity import (
    MIN_DISTANCE_M,
    ConditionalGravityBaseline,
    DistanceDecayBaseline,
    GravityBaseline,
    GravityBinnedBaseline,
)
from activitygraphs.ml.popularity import home_excluded_visit_rate

NUM_LOCATION_FEATURES = 1
IS_HOME_IDX = 0


def make_graphs(labels: torch.Tensor, homes: torch.Tensor, distances: torch.Tensor) -> list[pyg.data.Data]:
    """One graph per user, with ``is_home`` as the only per-user node feature and home distances in metres."""
    graphs = []
    for user_labels, home, user_distances in zip(labels, homes, distances, strict=True):
        n_nodes = len(user_labels)
        x_user_node = torch.zeros(n_nodes, 1)
        x_user_node[home, IS_HOME_IDX] = 1.0
        x_loc = torch.zeros(n_nodes, NUM_LOCATION_FEATURES)
        y = user_labels.float().unsqueeze(-1)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        graph = pyg.data.Data(x_loc=x_loc, x_user_node=x_user_node, edge_index=edge_index, y=y)
        graph.user_attr = torch.zeros(1, 1)
        graph.distances_m = user_distances.reshape(-1, 1)
        graphs.append(graph)
    return graphs


def simulate(
    n_users: int, n_nodes: int, beta_log: float, rate_per_m: float, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw labels from a gravity model with equal popularity everywhere; each user's home is at distance 0."""
    generator = torch.Generator().manual_seed(seed)
    homes = torch.randint(0, n_nodes, (n_users,), generator=generator)
    distances = 200.0 + 19_800.0 * torch.rand(n_users, n_nodes, generator=generator)
    distances[torch.arange(n_users), homes] = 0.0

    logits = 2.0 - beta_log * distances.clamp(min=MIN_DISTANCE_M).log() - rate_per_m * distances
    labels = (torch.rand(n_users, n_nodes, generator=generator) < logits.sigmoid()).long()
    labels[torch.arange(n_users), homes] = 1

    return labels, homes, distances


def loader_of(graphs: list[pyg.data.Data], batch_size: int = 64) -> pyg.loader.DataLoader:
    return pyg.loader.DataLoader(graphs, batch_size=batch_size)


class TestDistanceDecay:
    def test_ranks_nearest_first(self):
        distances = torch.tensor([[0.0, 900.0, 300.0, 5000.0, 1200.0]])
        batch = pyg.data.Batch.from_data_list(make_graphs(torch.zeros(1, 5), torch.tensor([0]), distances))

        scores = DistanceDecayBaseline()(batch).squeeze(-1)

        assert scores.argsort(descending=True).tolist() == [0, 2, 1, 4, 3]
        assert scores.sigmoid().unique().numel() == 5


class TestGravity:
    def test_the_tanner_fit_recovers_the_simulated_decay(self):
        """``beta_log`` is scale-free; ``beta_lin`` is the per-metre rate expressed in units of ``d0``."""
        beta_log, rate_per_m = 0.8, 1e-4
        labels, homes, distances = simulate(n_users=3000, n_nodes=40, beta_log=beta_log, rate_per_m=rate_per_m)

        gravity = GravityBaseline(IS_HOME_IDX).fit(loader_of(make_graphs(labels, homes, distances)))

        assert gravity.beta_log == pytest.approx(beta_log, abs=0.1)
        assert gravity.beta_lin == pytest.approx(rate_per_m * gravity.d0, abs=0.1)

    def test_d0_is_the_median_visited_non_home_distance(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=0.0)

        gravity = GravityBaseline(IS_HOME_IDX, max_iter=1).fit(loader_of(make_graphs(labels, homes, distances)))

        not_home = torch.ones_like(labels, dtype=torch.bool)
        not_home[torch.arange(len(labels)), homes] = False
        visited = distances[not_home & (labels > 0)]
        assert gravity.d0 == pytest.approx(float(visited.median()), rel=1e-3)

    def test_forward_is_the_pair_logit(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        graphs = make_graphs(labels, homes, distances)
        gravity = GravityBaseline(IS_HOME_IDX).fit(loader_of(graphs))

        batch = pyg.data.Batch.from_data_list(graphs[:2])
        out = gravity(batch).squeeze(-1)

        node = 3
        relative = max(float(batch.distances_m[node]), MIN_DISTANCE_M) / gravity.d0
        expected = (
            gravity.alpha
            + float(gravity.popularity_logit[node])
            - gravity.beta_log * np.log(relative)
            - gravity.beta_lin * relative
        )
        assert float(out[node]) == pytest.approx(expected, rel=1e-4)


class TestGravityBinned:
    def test_deterrence_is_observed_over_expected_per_bin(self):
        labels, homes, distances = simulate(n_users=300, n_nodes=12, beta_log=1.0, rate_per_m=0.0)
        graphs = make_graphs(labels, homes, distances)

        binned = GravityBinnedBaseline(IS_HOME_IDX, n_bins=4).fit(loader_of(graphs))

        split = extract_per_user_tensors(loader_of(graphs), IS_HOME_IDX)
        popularity = home_excluded_visit_rate(split.labels, split.home_idx)
        pair_distances = split.distances.clamp(min=MIN_DISTANCE_M)
        observed, expected = np.zeros(4), np.zeros(4)
        for user in range(len(labels)):
            for node in range(labels.shape[1]):
                if node == int(split.home_idx[user]):
                    continue
                b = int((pair_distances[user, node] > binned.edges).sum())
                observed[b] += float(split.labels[user, node])
                expected[b] += float(popularity[node])

        np.testing.assert_allclose(binned.log_deterrence.numpy(), np.log((observed + 0.5) / expected), rtol=1e-4)

    def test_one_finite_deterrence_per_bin(self):
        labels, homes, distances = simulate(n_users=300, n_nodes=12, beta_log=1.0, rate_per_m=0.0)

        binned = GravityBinnedBaseline(IS_HOME_IDX, n_bins=4).fit(loader_of(make_graphs(labels, homes, distances)))

        assert binned.edges.shape == (3,)
        assert binned.log_deterrence.shape == (4,)
        assert torch.all(binned.log_deterrence.isfinite())


class TestConditionalGravity:
    def test_an_unseen_home_is_scored_with_the_gravity_prior(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        homes = homes % 9
        graphs = make_graphs(labels, homes, distances)
        gravity = GravityBaseline(IS_HOME_IDX).fit(loader_of(graphs))
        conditional = ConditionalGravityBaseline(gravity).fit(loader_of(graphs[:150]), loader_of(graphs[150:]))

        unseen_distances = 200.0 + 1000.0 * torch.rand(1, 10)
        unseen_distances[0, 9] = 0.0
        unseen = pyg.data.Batch.from_data_list(make_graphs(torch.zeros(1, 10), torch.tensor([9]), unseen_distances))

        prior = gravity(unseen)
        out = conditional(unseen)

        non_home = slice(0, 9)
        torch.testing.assert_close(out[non_home], prior[non_home], rtol=1e-4, atol=1e-4)

    def test_the_strength_is_fitted(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        graphs = make_graphs(labels, homes, distances)
        gravity = GravityBaseline(IS_HOME_IDX).fit(loader_of(graphs))

        conditional = ConditionalGravityBaseline(gravity).fit(loader_of(graphs[:150]), loader_of(graphs[150:]))

        assert conditional.strength is not None
        assert conditional.counts.shape == (10, 10)
