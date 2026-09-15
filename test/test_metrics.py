"""Tests for the fully-connected baseline, hop-distance, and in-module hop-band ranking metrics."""

import pytest
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.metrics import (
    DEFAULT_HOP_BANDS,
    HopBandMetrics,
    PerUserRanking,
    RetrievalRPrecision,
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


class TestRetrievalRPrecisionUserCounts:
    """Home exclusion empties some users' ``RG_i``, so the average silently covers fewer users."""

    def _updated(self) -> RetrievalRPrecision:
        """Three users; the third has no positive left, as a home-only realised set does."""
        metric = RetrievalRPrecision()
        preds = torch.tensor([0.9, 0.1, 0.2, 0.9, 0.1, 0.3, 0.5, 0.4, 0.2])
        target = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0])
        indexes = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        metric.update(preds, target, indexes)
        return metric

    def test_counts_the_users_the_average_covers(self):
        assert self._updated().user_counts()["n_scored_users"] == 2

    def test_counts_the_users_with_nothing_left_to_rank(self):
        assert self._updated().user_counts()["n_dropped_users"] == 1

    def test_the_dropped_users_do_not_enter_the_average(self):
        """One perfect user and one bottom-ranked user average to 0.5, not to 1/3 over all three."""
        assert self._updated().compute().item() == pytest.approx(0.5)


def nothing_excluded(n: int) -> torch.Tensor:
    """An exclusion mask that drops no node, for the cases that are not about the home exclusion."""
    return torch.zeros(n, dtype=torch.bool)


def chain_hops(num_users: int, num_nodes: int) -> torch.Tensor:
    """Hops from home of ``num_users`` chain graphs laid end to end: node ``i`` of each is ``i`` hops away."""
    return torch.arange(num_nodes, dtype=torch.float).repeat(num_users)


class TestPerUserRanking:
    def _updated(self) -> PerUserRanking:
        """Two users with 2 positives each: user 0 ranks both first, user 1 ranks only one in its top 2."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        metric.update(
            preds,
            target,
            indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            exclude=nothing_excluded(8),
            hops=chain_hops(2, 4),
        )
        return metric

    def test_no_score_vectors_by_default(self):
        assert self._updated().score_vectors().numel() == 0

    def test_score_vectors_are_row_aligned_with_the_summary_columns(self):
        metric = PerUserRanking(k=2, store_score_vectors=True)
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        metric.update(
            preds,
            target,
            indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            exclude=nothing_excluded(8),
            hops=chain_hops(2, 4),
        )

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
        metric.update(
            preds, target, indexes=torch.tensor([0, 0, 1, 1]), exclude=nothing_excluded(4), hops=chain_hops(2, 2)
        )

        assert metric.columns()["user_id"].tolist() == [0]
        assert metric.score_vectors().shape == (1, 2)

    def test_retains_one_row_per_user(self):
        columns = self._updated().columns()

        assert set(columns) == {"user_id", "n_pos", "n_pos_home_incl", "r_precision", "recall"}
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

    def test_home_node_is_neither_a_candidate_nor_a_positive(self):
        """One user whose top-ranked node is their home: dropping it costs them the free hit."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        is_home = torch.tensor([True, False, False, False])
        metric.update(preds, target, indexes=torch.zeros(4, dtype=torch.long), exclude=is_home, hops=chain_hops(1, 4))

        columns = metric.columns()

        assert columns["n_pos"].tolist() == [1]
        assert columns["n_pos_home_incl"].tolist() == [2]
        # Only node 1 is left as a positive, and it is ranked below node 2 among the candidates.
        assert columns["r_precision"].tolist() == pytest.approx([0.0])
        assert columns["recall"].tolist() == pytest.approx([1.0])

    def test_a_home_only_user_is_dropped(self):
        """A user whose sole visit is their own home has an empty target once home is excluded."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 0, 0, 0, 1, 1, 0, 0])
        is_home = torch.tensor([True, False, False, False, True, False, False, False])
        metric.update(
            preds,
            target,
            indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            exclude=is_home,
            hops=chain_hops(2, 4),
        )

        assert metric.columns()["user_id"].tolist() == [1]

    def test_the_retained_score_vector_still_covers_the_home_node(self):
        """The home-rank health check reads these vectors, so the excluded node's score must survive."""
        metric = PerUserRanking(k=2, store_score_vectors=True)
        preds = torch.tensor([0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        is_home = torch.tensor([True, False, False, False])
        metric.update(preds, target, indexes=torch.zeros(4, dtype=torch.long), exclude=is_home, hops=chain_hops(1, 4))

        assert metric.score_vectors()[0].tolist() == pytest.approx([0.9, 0.2, 0.8, 0.1])

    def test_skips_users_without_positives(self):
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.1, 0.9, 0.1])
        target = torch.tensor([1, 0, 0, 0])  # user 1 has no positives
        metric.update(
            preds, target, indexes=torch.tensor([0, 0, 1, 1]), exclude=nothing_excluded(4), hops=chain_hops(2, 2)
        )

        assert metric.columns()["user_id"].tolist() == [0]

    def test_accumulates_across_batches(self):
        metric = self._updated()
        preds = torch.tensor([0.9, 0.1])
        metric.update(
            preds,
            torch.tensor([1, 0]),
            indexes=torch.tensor([2, 2]),
            exclude=nothing_excluded(2),
            hops=chain_hops(1, 2),
        )

        assert metric.columns()["user_id"].tolist() == [0, 1, 2]

    def test_each_positive_keeps_its_hop_distance(self):
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
        hops = torch.tensor([0.0, 4.0, 7.0, float("inf"), 1.0, 5.0, 9.0, 13.0])
        metric.update(
            preds, target, indexes=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), exclude=nothing_excluded(8), hops=hops
        )

        assert metric.positive_columns()["pos_hops"] == [[4.0, float("inf")], [1.0, 13.0]]

    def test_flags_mark_the_positives_ranked_inside_the_cutoffs(self):
        positives = self._updated().positive_columns()

        assert positives["pos_in_top_r"] == [[True, True], [True, False]]
        assert positives["pos_in_top_k"] == [[True, True], [True, False]]

    def test_flags_reproduce_the_per_user_scores(self):
        """Summing a flag over a user's positives and dividing by ``n_pos`` gives back the retained score."""
        metric = self._updated()
        columns = metric.columns()
        positives = metric.positive_columns()

        for row, n_pos in enumerate(columns["n_pos"].tolist()):
            assert sum(positives["pos_in_top_r"][row]) / n_pos == pytest.approx(columns["r_precision"][row].item())
            assert sum(positives["pos_in_top_k"][row]) / n_pos == pytest.approx(columns["recall"][row].item())

    def test_counts_the_candidates_scored_above_each_positive(self):
        positives = self._updated().positive_columns()

        assert positives["pos_n_scored_higher"] == [[0, 1], [0, 2]]
        assert positives["pos_n_tied"] == [[0, 0], [0, 0]]

    def test_counts_reproduce_the_flags_without_ties(self):
        metric = self._updated()
        n_pos = metric.columns()["n_pos"].tolist()
        positives = metric.positive_columns()

        for row, n_scored_higher in enumerate(positives["pos_n_scored_higher"]):
            assert [n < n_pos[row] for n in n_scored_higher] == positives["pos_in_top_r"][row]
            assert [n < metric.k for n in n_scored_higher] == positives["pos_in_top_k"][row]

    def test_the_excluded_home_node_is_not_counted_above_a_positive(self):
        """Home scores highest but is not a candidate, so only node 2 scores above the remaining positive."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        is_home = torch.tensor([True, False, False, False])
        metric.update(preds, target, indexes=torch.zeros(4, dtype=torch.long), exclude=is_home, hops=chain_hops(1, 4))

        assert metric.positive_columns()["pos_n_scored_higher"] == [[1]]

    def test_ties_are_counted_apart_from_higher_scores(self):
        """Node 0 scores above both positives; each positive ties with node 2 and with the other positive."""
        metric = PerUserRanking(k=2)
        metric.update(
            torch.tensor([0.9, 0.5, 0.5, 0.5, 0.1]),
            torch.tensor([0, 1, 0, 1, 0]),
            indexes=torch.zeros(5, dtype=torch.long),
            exclude=nothing_excluded(5),
            hops=chain_hops(1, 5),
        )

        positives = metric.positive_columns()

        assert positives["pos_n_scored_higher"] == [[1, 1]]
        assert positives["pos_n_tied"] == [[2, 2]]

    def test_the_home_node_is_not_a_retained_positive(self):
        """Only node 1 is left as a positive: outside the top-1, inside the top-2."""
        metric = PerUserRanking(k=2)
        preds = torch.tensor([0.9, 0.2, 0.8, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        is_home = torch.tensor([True, False, False, False])
        metric.update(preds, target, indexes=torch.zeros(4, dtype=torch.long), exclude=is_home, hops=chain_hops(1, 4))

        positives = metric.positive_columns()

        assert positives["pos_hops"] == [[1.0]]
        assert positives["pos_in_top_r"] == [[False]]
        assert positives["pos_in_top_k"] == [[True]]

    def test_positive_columns_stay_row_aligned_across_batches(self):
        metric = self._updated()
        metric.update(
            torch.tensor([0.9, 0.1, 0.5]),
            torch.tensor([0, 1, 1]),
            indexes=torch.tensor([2, 2, 2]),
            exclude=nothing_excluded(3),
            hops=torch.tensor([0.0, 1.0, 6.0]),
        )

        positives = metric.positive_columns()

        assert [len(hops) for hops in positives["pos_hops"]] == metric.columns()["n_pos"].tolist()
        assert positives["pos_hops"][2] == [1.0, 6.0]
        assert positives["pos_n_scored_higher"][2] == [2, 1]


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
            f"test_hop_3-5_recall@{k}",
            f"test_hop_3-5_ndcg@{k}",
            "test_hop_3-5_nll",
            "test_hop_3-5_n_pos",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)
        # node 0 (home) was the only positive in band 0-2; excluding it empties the band
        assert not any(key.startswith("test_hop_0-2") for key in logged)

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
        # Band 0-2 held only the home node's positive, so it no longer reports a row.
        assert [row["hop_band"] for row in rows] == ["3-5"]
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
        assert set(columns) == {
            "user_id",
            "n_pos",
            "n_pos_home_incl",
            "r_precision",
            "recall",
            "pos_hops",
            "pos_in_top_r",
            "pos_in_top_k",
            "pos_n_scored_higher",
            "pos_n_tied",
        }
        assert columns["user_id"] == [0]
        assert columns["n_pos"] == [1]  # two positives, one of which is the excluded home node
        assert columns["n_pos_home_incl"] == [2]
        assert 0.0 <= columns["r_precision"][0] <= 1.0
        assert columns["pos_hops"] == [[3.0]]
