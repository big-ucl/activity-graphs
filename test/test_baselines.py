"""Tests for frequency-based baseline models."""

import torch
import torch_geometric as pyg
import pytest

from activitygraphs.ml.baselines import (
    NodeBaseline,
    ConditionalNodeBaseline,
    compute_ptr_from_batch,
)


def make_graphs(num_graphs: int, num_nodes: int, num_features: int, seed: int = 0) -> list[pyg.data.Data]:
    """Return a list of random graphs with the given dimensions."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        x = torch.rand(num_nodes, num_features, generator=rng)
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        graphs.append(pyg.data.Data(x=x, edge_index=edge_index, y=y))
    return graphs


@pytest.fixture
def batch_uniform():
    """Batched PyG data: 4 graphs, each with 5 nodes."""
    graphs = make_graphs(num_graphs=4, num_nodes=5, num_features=3)
    return pyg.data.Batch.from_data_list(graphs)


@pytest.fixture
def batch_variable():
    """Batched PyG data: graphs with different numbers of nodes (3, 5, 2)."""
    rng = torch.Generator().manual_seed(42)
    graphs = []
    for n in [3, 5, 2]:
        x = torch.rand(n, 3, generator=rng)
        y = torch.randint(0, 2, (n, 1), generator=rng).float()
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        graphs.append(pyg.data.Data(x=x, edge_index=edge_index, y=y))
    return pyg.data.Batch.from_data_list(graphs)


class TestComputePtrFromBatch:
    def test_matches_pyg_ptr_uniform(self, batch_uniform):
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert torch.equal(computed, batch_uniform.ptr)

    def test_matches_pyg_ptr_variable(self, batch_variable):
        computed = compute_ptr_from_batch(batch_variable.batch)
        assert torch.equal(computed, batch_variable.ptr)

    def test_single_graph(self):
        batch = torch.zeros(7, dtype=torch.long)
        computed = compute_ptr_from_batch(batch)
        assert torch.equal(computed, torch.tensor([0, 7]))

    def test_ptr_length(self, batch_uniform):
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert len(computed) == batch_uniform.num_graphs + 1

    def test_ptr_starts_at_zero(self, batch_uniform):
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert computed[0].item() == 0

    def test_ptr_ends_at_num_nodes(self, batch_uniform):
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert computed[-1].item() == batch_uniform.num_nodes


class TestNodeBaseline:
    def test_node_indices_match_within_graph_position(self, batch_uniform):
        """Each node's inferred index should equal its position within its graph."""
        ptr = compute_ptr_from_batch(batch_uniform.batch)
        node_indices = torch.arange(batch_uniform.num_nodes) - ptr[batch_uniform.batch]
        expected = torch.arange(batch_uniform.num_nodes) % 5  # uniform graphs of size 5
        assert torch.equal(node_indices, expected)

    def test_forward_shape(self, batch_uniform):
        num_nodes = 5
        baseline = NodeBaseline(num_nodes=num_nodes)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert out.shape == (batch_uniform.num_nodes, 1)

    def test_forward_values_in_logit_range(self, batch_uniform):
        num_nodes = 5
        baseline = NodeBaseline(num_nodes=num_nodes)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert torch.isfinite(out).all()


class TestConditionalNodeBaseline:
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

        baseline = ConditionalNodeBaseline(num_nodes=num_nodes, is_home_idx=is_home_idx)
        baseline.fit(loader)
        out = baseline(batched.x, batched.edge_index, batch=batched.batch)
        assert out.shape == (batched.num_nodes, 1)
