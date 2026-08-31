"""Tests for the fully-connected baseline, hop-distance, and in-module hop-band ranking metrics."""

import pytest
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.metrics import (
    DEFAULT_HOP_BANDS,
    HopBandMetrics,
    PerUserRanking,
    compute_home_hop_distance,
    hop_band_scalars,
)
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


class TestHopBandMetricsStandalone:
    def _updated(self) -> HopBandMetrics:
        """Two users, positives at hop 1 (band 0-2) and hop 4 (band 3-5); band 6-8 sees no positive."""
        metrics = HopBandMetrics(DEFAULT_HOP_BANDS, k=2)
        hop = torch.tensor([0.0, 1.0, 4.0, 7.0, 0.0, 1.0, 4.0, 7.0])
        logits = torch.tensor([0.1, 2.0, 2.0, 0.1, 0.1, 2.0, 2.0, 0.1])
        target = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
        metrics.update(hop, logits, target, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
        return metrics

    def test_rows_skip_bands_without_positives(self):
        rows = self._updated().compute()

        assert [row["hop_band"] for row in rows] == ["0-2", "3-5"]
        assert [row["hop_low"] for row in rows] == [0, 3]
        assert all(row["k"] == 2 for row in rows)
        assert rows[0]["recall"] == pytest.approx(1.0)  # the sole positive is top-ranked in its band
        assert rows[0]["n_pos"] == 2.0  # one positive per user

    def test_reset_clears_accumulated_state(self):
        metrics = self._updated()
        metrics.reset()

        assert metrics.compute() == []

    def test_scalars_keep_the_at_k_suffix_on_ranking_metrics_only(self):
        scalars = hop_band_scalars(self._updated().compute())

        assert set(scalars) == {
            "test_hop_0-2_recall@2",
            "test_hop_0-2_ndcg@2",
            "test_hop_0-2_nll",
            "test_hop_0-2_n_pos",
            "test_hop_3-5_recall@2",
            "test_hop_3-5_ndcg@2",
            "test_hop_3-5_nll",
            "test_hop_3-5_n_pos",
        }


class TestPerUserRanking:
    def _updated(self) -> PerUserRanking:
        """Two users with 2 positives each: user 0 ranks both first, user 1 ranks only one in its top 2."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        metric.update(preds, target, indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
        return metric

    def test_no_score_vectors_by_default(self):
        assert self._updated().score_vectors().numel() == 0

    def test_score_vectors_are_row_aligned_with_the_summary_columns(self):
        metric = PerUserRanking(k=2, store_score_vectors=True)
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        metric.update(preds, target, indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))

        vectors = metric.score_vectors()

        assert vectors.shape == (2, 4)
        assert metric.columns()["user_id"].tolist() == [0, 1]
        assert vectors[0].tolist() == pytest.approx([0.9, 0.8, 0.2, 0.1])
        assert vectors[1].tolist() == pytest.approx([0.9, 0.2, 0.8, 0.1])

    def test_score_vectors_skip_users_without_positives(self):
        """A user with no positives is dropped from the summary rows; its vector must be dropped too."""
        metric = PerUserRanking(k=2, store_score_vectors=True)
        preds = torch.tensor([0.9, 0.1, 0.5, 0.4])
        target = torch.tensor([1, 0, 0, 0])
        metric.update(preds, target, indexes=torch.tensor([0, 0, 1, 1]))

        assert metric.columns()["user_id"].tolist() == [0]
        assert metric.score_vectors().shape == (1, 2)

    def test_retains_one_row_per_user(self):
        columns = self._updated().columns()

        assert set(columns) == {"user_id", "n_pos", "r_precision", "recall"}
        assert all(len(column) == 2 for column in columns.values())
        assert columns["user_id"].tolist() == [0, 1]
        assert columns["n_pos"].tolist() == [2, 2]

    def test_scores_match_the_hand_computed_values(self):
        columns = self._updated().columns()

        # user 0: both positives inside the top 2; user 1: one of two.
        assert columns["r_precision"].tolist() == pytest.approx([1.0, 0.5])
        assert columns["recall"].tolist() == pytest.approx([1.0, 0.5])

    def test_aggregate_is_the_mean_of_the_retained_scores(self):
        columns = self._updated().columns()

        assert columns["r_precision"].mean().item() == pytest.approx(0.75)

    def test_skips_users_without_positives(self):
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.1, 0.9, 0.1])
        target = torch.tensor([1, 0, 0, 0])  # user 1 has no positives
        metric.update(preds, target, indexes=torch.tensor([0, 0, 1, 1]))

        assert metric.columns()["user_id"].tolist() == [0]

    def test_accumulates_across_batches(self):
        metric = self._updated()
        preds = torch.tensor([0.9, 0.1])
        metric.update(preds, torch.tensor([1, 0]), indexes=torch.tensor([2, 2]))

        assert metric.columns()["user_id"].tolist() == [0, 1, 2]

    def test_reset_clears_accumulated_state(self):
        metric = self._updated()
        metric.reset()

        assert metric.columns()["user_id"].numel() == 0


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
            "test_hop_0-2_nll",
            "test_hop_0-2_n_pos",
            "test_hop_3-5_nll",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)
        # node 0 (home) is the only positive in band 0-2
        assert torch.as_tensor(logged["test_hop_0-2_n_pos"]).item() == 1.0

    def test_hop_band_rows_are_long_format(self, monkeypatch):
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(
            model=model,
            lr=1e-3,
            pos_weight=torch.tensor(1.0),
            home_hop_distance=hop,
            is_home_idx=0,
        )

        monkeypatch.setattr(module, "log", lambda key, val, **kw: None)
        monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: None)

        batch = pyg.data.Batch.from_data_list([self._make_graph()])
        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        rows = module.hop_band_rows
        assert [row["hop_band"] for row in rows] == ["0-2", "3-5"]
        assert all(set(row) == {"hop_band", "hop_low", "k", "recall", "ndcg", "nll", "n_pos"} for row in rows)
        assert all(isinstance(row["ndcg"], float) for row in rows)
        assert rows[0]["n_pos"] == 1.0

    def test_per_user_columns_populated_at_test_end(self, monkeypatch):
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(
            model=model,
            lr=1e-3,
            pos_weight=torch.tensor(1.0),
            home_hop_distance=hop,
            is_home_idx=0,
        )

        monkeypatch.setattr(module, "log", lambda key, val, **kw: None)
        monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: None)

        batch = pyg.data.Batch.from_data_list([self._make_graph()])
        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        columns = module.per_user_columns
        assert set(columns) == {"user_id", "n_pos", "r_precision", "recall"}
        assert columns["user_id"] == [0]
        assert columns["n_pos"] == [2]  # the graph has two positives
        assert 0.0 <= columns["r_precision"][0] <= 1.0

    def test_no_hop_band_metrics_without_hop_distance(self):
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(model=model, loss=None, lr=1e-3, pos_weight=torch.tensor(1.0))
        assert module.hop_band_metrics is None
