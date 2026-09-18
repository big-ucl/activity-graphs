"""Tests for frequency-based baseline models."""

import torch
import torch_geometric as pyg
import pytest

from activitygraphs.ml.baselines import ConditionalVisitFrequencyBaseline, VisitFrequencyBaseline
from activitygraphs.ml.baselines.common import (
    PerUserTensors,
    extract_per_user_tensors,
    fit_smoothing_strength,
    smoothed_rate,
)
from activitygraphs.ml.popularity import home_excluded_visit_rate, home_visit_counts

IS_HOME_IDX = 2


def make_graphs(num_graphs: int, num_nodes: int, num_features: int, seed: int = 0) -> list[pyg.data.Data]:
    """Return random graphs whose ``IS_HOME_IDX`` column marks one home node per graph, with home distances."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        x = torch.rand(num_nodes, num_features, generator=rng)
        x[:, IS_HOME_IDX] = 0.0
        x[int(torch.randint(0, num_nodes, (1,), generator=rng)), IS_HOME_IDX] = 1.0
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        graph = pyg.data.Data(x=x, edge_index=edge_index, y=y)
        graph.distances = torch.rand(num_nodes, 1, generator=rng)
        graphs.append(graph)
    return graphs


@pytest.fixture
def batch_uniform():
    """Batched PyG data: 4 graphs, each with 5 nodes."""
    graphs = make_graphs(num_graphs=4, num_nodes=5, num_features=3)
    return pyg.data.Batch.from_data_list(graphs)


class TestStackSplit:
    def test_rows_follow_the_loader_order(self):
        graphs = make_graphs(num_graphs=7, num_nodes=5, num_features=3)
        split = extract_per_user_tensors(pyg.loader.DataLoader(graphs, batch_size=3), IS_HOME_IDX)

        assert split.labels.shape == (7, 5)
        assert split.labels.tolist() == [g.y.squeeze(-1).tolist() for g in graphs]
        assert split.home_idx.tolist() == [int(g.x[:, IS_HOME_IDX].argmax()) for g in graphs]
        assert split.distances.tolist() == [g.distances.squeeze(-1).tolist() for g in graphs]

    def test_a_graph_without_a_home_raises(self):
        graphs = make_graphs(num_graphs=2, num_nodes=5, num_features=3)
        graphs[1].x[:, IS_HOME_IDX] = 0.0

        with pytest.raises(ValueError, match="No home node"):
            extract_per_user_tensors(pyg.loader.DataLoader(graphs, batch_size=2), IS_HOME_IDX)


class TestVisitFrequencyBaseline:
    def test_forward_shape(self, batch_uniform):
        num_nodes = 5
        baseline = VisitFrequencyBaseline(num_nodes=num_nodes, is_home_idx=IS_HOME_IDX)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert out.shape == (batch_uniform.num_nodes, 1)

    def test_forward_values_in_logit_range(self, batch_uniform):
        num_nodes = 5
        baseline = VisitFrequencyBaseline(num_nodes=num_nodes, is_home_idx=IS_HOME_IDX)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert torch.isfinite(out).all()

    def test_scores_are_the_home_excluded_visit_rate(self, batch_uniform):
        graphs = make_graphs(8, 5, 3)
        baseline = VisitFrequencyBaseline(num_nodes=5, is_home_idx=IS_HOME_IDX)
        baseline.fit(pyg.loader.DataLoader(graphs, batch_size=4))

        split = extract_per_user_tensors(pyg.loader.DataLoader(graphs, batch_size=8), IS_HOME_IDX)
        expected = torch.logit(home_excluded_visit_rate(split.labels, split.home_idx))
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)

        torch.testing.assert_close(out.squeeze(-1), expected.repeat(batch_uniform.num_graphs))


class TestConditionalVisitFrequencyBaseline:
    def test_forward_shape(self, batch_uniform):
        num_nodes = 5
        # Use a synthetic feature matrix where a chosen column acts as the home indicator.
        is_home_idx = 2
        num_features = 3
        graphs = make_graphs(num_graphs=8, num_nodes=num_nodes, num_features=num_features)
        for g in graphs:
            g.x[:, is_home_idx] = 0.0
            g.x[0, is_home_idx] = 1.0  # mark node 0 as home

        loader = pyg.loader.DataLoader(graphs, batch_size=4)
        batched = pyg.data.Batch.from_data_list(graphs[:4])

        baseline = ConditionalVisitFrequencyBaseline(num_nodes=num_nodes, is_home_idx=is_home_idx)
        baseline.fit(loader, loader)
        out = baseline(batched.x, batched.edge_index, batch=batched.batch)
        assert out.shape == (batched.num_nodes, 1)


def split_of(labels: list[list[int]], homes: list[int]) -> PerUserTensors:
    labels_tensor = torch.tensor(labels)
    return PerUserTensors(
        labels=labels_tensor, home_idx=torch.tensor(homes), distances=torch.zeros_like(labels_tensor, dtype=torch.float)
    )


def graphs_of(split: PerUserTensors) -> list[pyg.data.Data]:
    """One graph per user of ``split``, with home and labels as given and zero distances."""
    graphs = []
    for labels, home in zip(split.labels, split.home_idx, strict=True):
        x = torch.zeros(len(labels), IS_HOME_IDX + 1)
        x[home, IS_HOME_IDX] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=labels.float().unsqueeze(-1))
        graph.distances = torch.zeros(len(labels), 1)
        graphs.append(graph)
    return graphs


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
        counts, users = home_visit_counts(train.labels, train.home_idx)

        strength = fit_smoothing_strength(counts, users, val, torch.full((6, 3), 0.5))

        assert strength == pytest.approx(0.01)

    def test_strength_is_large_when_validation_follows_the_prior(self):
        """Training residents of home 0 look home-specific, but validation residents visit at the prior rate."""
        train = split_of([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], [0, 0, 0, 0])
        val = split_of([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], [0, 0, 0, 0])
        counts, users = home_visit_counts(train.labels, train.home_idx)

        strength = fit_smoothing_strength(counts, users, val, torch.full((4, 3), 0.5))

        assert strength == pytest.approx(1000.0)


class TestConditionalVisitFrequencySmoothing:
    def test_an_unseen_home_is_scored_with_the_visit_frequency(self):
        train = split_of([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], [0, 0, 1])
        val = split_of([[1, 1, 0, 1], [0, 1, 0, 0]], [0, 1])
        conditional = ConditionalVisitFrequencyBaseline(num_nodes=4, is_home_idx=IS_HOME_IDX).fit(
            pyg.loader.DataLoader(graphs_of(train), batch_size=2), pyg.loader.DataLoader(graphs_of(val), batch_size=2)
        )
        frequency = VisitFrequencyBaseline(num_nodes=4, is_home_idx=IS_HOME_IDX).fit(
            pyg.loader.DataLoader(graphs_of(train), batch_size=2)
        )

        unseen = pyg.data.Batch.from_data_list(graphs_of(split_of([[0, 0, 0, 1]], [3])))
        args = (unseen.x, unseen.edge_index)

        torch.testing.assert_close(conditional(*args, batch=unseen.batch), frequency(*args, batch=unseen.batch))

    def test_the_strength_is_fitted(self):
        train = split_of([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], [0, 0, 1])
        loader = pyg.loader.DataLoader(graphs_of(train), batch_size=2)

        conditional = ConditionalVisitFrequencyBaseline(num_nodes=4, is_home_idx=IS_HOME_IDX).fit(loader, loader)

        assert conditional.strength is not None
        assert conditional.logits.shape == (4, 4)
