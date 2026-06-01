"""Tests for the fully-connected baseline, hop-distance, and in-module hop-band ranking metrics."""

import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.metrics import compute_home_hop_distance
from activitygraphs.ml.models import FullyConnectedMLP, NodeMLP

# Path graph 0-1-2-3-4 (both directions): hop distance from node 0 is [0, 1, 2, 3, 4].
PATH_EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]])


class TestFullyConnectedMLP:
    def test_forward_shape(self):
        num_nodes, in_features, num_graphs = 4, 3, 2
        model = FullyConnectedMLP(num_nodes, in_features, hidden_channels=8, num_layers=3, dropout=0.0)
        x = torch.rand(num_graphs * num_nodes, in_features)
        out = model(x, torch.zeros(2, 0, dtype=torch.long), None, None)
        assert out.shape == (num_graphs * num_nodes, 1)

    def test_rejects_wrong_node_count(self):
        model = FullyConnectedMLP(num_nodes=4, in_features=3, hidden_channels=8)
        x = torch.rand(5, 3)  # not a multiple of 4
        try:
            model(x, torch.zeros(2, 0, dtype=torch.long), None, None)
            raise AssertionError("expected a ValueError for a non-multiple node count")
        except ValueError:
            pass


class TestHomeHopDistance:
    def test_path_graph(self):
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        assert hop[0].tolist() == [0, 1, 2, 3, 4]
        assert hop[2].tolist() == [2, 1, 0, 1, 2]

    def test_disconnected_is_inf(self):
        edge_index = torch.tensor([[0], [1]])  # only 0-1 connected; node 2 isolated
        hop = compute_home_hop_distance(edge_index, num_nodes=3)
        assert hop[0, 2] == float("inf")


class TestHopBandMetrics:
    def _make_graph(self):
        """One 5-node graph; node 0 is home, positives at hop 0 (band 0-2) and hop 3 (band 3-5)."""
        x = torch.zeros(5, 2)
        x[0, 0] = 1.0  # is_home at column 0, node 0
        x[:, 1] = torch.rand(5)
        y = torch.zeros(5, 1)
        y[0] = 1.0
        y[3] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=y)
        graph.graph_x = torch.zeros(1, 2)  # demographics
        graph.user_id = torch.tensor([0])
        return graph

    def test_hop_band_keys_logged(self, monkeypatch):
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(
            model=model,
            lr=1e-3,
            pos_weight=torch.tensor(1.0),
            home_hop_distance=hop,
            is_home_idx=0,
        )

        logged: dict = {}
        monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.__setitem__(key, val))
        monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: logged.update(mapping))

        batch = pyg.data.Batch.from_data_list([self._make_graph()])
        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        k = module.k
        expected = {
            f"test_hop_0-2_recall@{k}",
            f"test_hop_0-2_ndcg@{k}",
            f"test_hop_3-5_recall@{k}",
            "test_hop_0-2_bce_weighted",
            "test_hop_0-2_n_pos",
            "test_hop_3-5_bce_weighted",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)
        # node 0 (home) is the only positive in band 0-2
        assert torch.as_tensor(logged["test_hop_0-2_n_pos"]).item() == 1.0

    def test_no_hop_band_metrics_without_hop_distance(self):
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(model=model, lr=1e-3, pos_weight=torch.tensor(1.0))
        assert module.hop_band_test_metrics is None
