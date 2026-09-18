"""Tests for the distance-decay baselines."""

import numpy as np
import pytest
import torch
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler

from activitygraphs.ml.baselines.common import extract_per_user_tensors
from activitygraphs.ml.baselines.gravity import (
    MIN_DISTANCE_M,
    ConditionalGravityBaseline,
    DistanceDecayBaseline,
    GravityBaseline,
    GravityBinnedBaseline,
    unscale_distances,
)
from activitygraphs.ml.lightning_module import extract_features
from activitygraphs.ml.popularity import home_excluded_visit_rate

IS_HOME_IDX = 0


def fit_scaler(distances: torch.Tensor) -> StandardScaler:
    return StandardScaler().fit(distances.reshape(-1, 1).numpy())


def make_graphs(
    labels: torch.Tensor, homes: torch.Tensor, distances: torch.Tensor, scaler: StandardScaler
) -> list[pyg.data.Data]:
    """One graph per user, with ``is_home`` in column 0 and scaled home distances, as the dataset stores them."""
    graphs = []
    for user_labels, home, user_distances in zip(labels, homes, distances, strict=True):
        n_nodes = len(user_labels)
        x = torch.zeros(n_nodes, 2)
        x[home, IS_HOME_IDX] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=user_labels.float().unsqueeze(-1))
        graph.graph_x = torch.zeros(1, 1)
        graph.distances = torch.from_numpy(scaler.transform(user_distances.reshape(-1, 1).numpy())).float()
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


class TestUnscaleDistances:
    def test_round_trips_the_scaler_and_clamps_the_home(self):
        distances = torch.tensor([[0.0, 500.0, 2500.0]])
        scaler = fit_scaler(distances)
        scaled = torch.from_numpy(scaler.transform(distances.reshape(-1, 1).numpy())).float().reshape(1, 3)

        restored = unscale_distances(scaled, scaler)

        assert restored[0].tolist() == pytest.approx([MIN_DISTANCE_M, 500.0, 2500.0], rel=1e-5)


class TestDistanceColumn:
    def test_full_info_appends_the_stored_distance_as_the_last_column(self):
        """The distance baselines read ``x[:, -1]``; this pins that ``extract_features`` puts the distance there."""
        labels, homes, distances = simulate(n_users=3, n_nodes=5, beta_log=1.0, rate_per_m=0.0)
        batch = pyg.data.Batch.from_data_list(make_graphs(labels, homes, distances, fit_scaler(distances)))

        x = extract_features(batch, full_info=True, is_home_idx=IS_HOME_IDX, use_demographics=True)

        torch.testing.assert_close(x[:, -1], batch.distances.squeeze(-1))


class TestDistanceDecay:
    def test_ranks_nearest_first(self):
        distances = torch.tensor([[0.0, 900.0, 300.0, 5000.0, 1200.0]])
        scaler = fit_scaler(distances)
        batch = pyg.data.Batch.from_data_list(make_graphs(torch.zeros(1, 5), torch.tensor([0]), distances, scaler))
        x = extract_features(batch, full_info=True, is_home_idx=IS_HOME_IDX)

        scores = DistanceDecayBaseline(scaler)(x, batch.edge_index, batch=batch.batch).squeeze(-1)

        assert scores.argsort(descending=True).tolist() == [0, 2, 1, 4, 3]
        assert scores.sigmoid().unique().numel() == 5


class TestGravity:
    def test_the_tanner_fit_recovers_the_simulated_decay(self):
        """``beta_log`` is scale-free; ``beta_lin`` is the per-metre rate expressed in units of ``d0``."""
        beta_log, rate_per_m = 0.8, 1e-4
        labels, homes, distances = simulate(n_users=3000, n_nodes=40, beta_log=beta_log, rate_per_m=rate_per_m)
        scaler = fit_scaler(distances)

        gravity = GravityBaseline(IS_HOME_IDX, scaler).fit(loader_of(make_graphs(labels, homes, distances, scaler)))

        assert gravity.beta_log == pytest.approx(beta_log, abs=0.1)
        assert gravity.beta_lin == pytest.approx(rate_per_m * gravity.d0, abs=0.1)

    def test_d0_is_the_median_visited_non_home_distance(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=0.0)
        scaler = fit_scaler(distances)

        gravity = GravityBaseline(IS_HOME_IDX, scaler, max_iter=1).fit(
            loader_of(make_graphs(labels, homes, distances, scaler))
        )

        not_home = torch.ones_like(labels, dtype=torch.bool)
        not_home[torch.arange(len(labels)), homes] = False
        visited = distances[not_home & (labels > 0)]
        assert gravity.d0 == pytest.approx(float(visited.median()), rel=1e-3)

    def test_forward_is_the_pair_logit(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        scaler = fit_scaler(distances)
        graphs = make_graphs(labels, homes, distances, scaler)
        gravity = GravityBaseline(IS_HOME_IDX, scaler).fit(loader_of(graphs))

        batch = pyg.data.Batch.from_data_list(graphs[:2])
        x = extract_features(batch, full_info=True, is_home_idx=IS_HOME_IDX)
        out = gravity(x, batch.edge_index, batch=batch.batch).squeeze(-1)

        node = 3
        relative = float(unscale_distances(batch.distances[node], scaler)) / gravity.d0
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
        scaler = fit_scaler(distances)
        graphs = make_graphs(labels, homes, distances, scaler)

        binned = GravityBinnedBaseline(IS_HOME_IDX, scaler, n_bins=4).fit(loader_of(graphs))

        split = extract_per_user_tensors(loader_of(graphs), IS_HOME_IDX)
        popularity = home_excluded_visit_rate(split.labels, split.home_idx)
        pair_distances = unscale_distances(split.distances, scaler)
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
        scaler = fit_scaler(distances)

        binned = GravityBinnedBaseline(IS_HOME_IDX, scaler, n_bins=4).fit(
            loader_of(make_graphs(labels, homes, distances, scaler))
        )

        assert binned.edges.shape == (3,)
        assert binned.log_deterrence.shape == (4,)
        assert torch.all(binned.log_deterrence.isfinite())


class TestConditionalGravity:
    def test_an_unseen_home_is_scored_with_the_gravity_prior(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        homes = homes % 9
        scaler = fit_scaler(distances)
        graphs = make_graphs(labels, homes, distances, scaler)
        gravity = GravityBaseline(IS_HOME_IDX, scaler).fit(loader_of(graphs))
        conditional = ConditionalGravityBaseline(gravity).fit(loader_of(graphs[:150]), loader_of(graphs[150:]))

        unseen_distances = 200.0 + 1000.0 * torch.rand(1, 10)
        unseen_distances[0, 9] = 0.0
        unseen = pyg.data.Batch.from_data_list(
            make_graphs(torch.zeros(1, 10), torch.tensor([9]), unseen_distances, scaler)
        )
        x = extract_features(unseen, full_info=True, is_home_idx=IS_HOME_IDX)

        prior = gravity(x, unseen.edge_index, batch=unseen.batch)
        out = conditional(x, unseen.edge_index, batch=unseen.batch)

        non_home = slice(0, 9)
        torch.testing.assert_close(out[non_home], prior[non_home], rtol=1e-4, atol=1e-4)

    def test_the_strength_is_fitted(self):
        labels, homes, distances = simulate(n_users=200, n_nodes=10, beta_log=0.5, rate_per_m=1e-4)
        scaler = fit_scaler(distances)
        graphs = make_graphs(labels, homes, distances, scaler)
        gravity = GravityBaseline(IS_HOME_IDX, scaler).fit(loader_of(graphs))

        conditional = ConditionalGravityBaseline(gravity).fit(loader_of(graphs[:150]), loader_of(graphs[150:]))

        assert conditional.strength is not None
        assert conditional.counts.shape == (10, 10)
