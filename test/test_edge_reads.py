"""Unit tests for the step 6 detection reads, on small hand-built ranks, embeddings and graphs."""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from activitygraphs.base import PER_USER_STAGE
from activitygraphs.models.networks import GAT

EDGE_READS_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "accessibility" / "edge_reads.py"
edge_reads_spec = importlib.util.spec_from_file_location("edge_reads", EDGE_READS_PATH)
assert edge_reads_spec is not None and edge_reads_spec.loader is not None
edge_reads = importlib.util.module_from_spec(edge_reads_spec)
edge_reads_spec.loader.exec_module(edge_reads)

GATProbe = edge_reads.GATProbe
arm_families = edge_reads.arm_families
grouping_model = edge_reads.grouping_model
attention_summary = edge_reads.attention_summary
embedding_similarity = edge_reads.embedding_similarity
grouping_ranks = edge_reads.grouping_ranks
paired_rank_tail = edge_reads.paired_rank_tail
paired_zone_split = edge_reads.paired_zone_split
positive_frame = edge_reads.positive_frame
rail_degree = edge_reads.rail_degree
zone_classes = edge_reads.zone_classes


class TestZoneClasses:
    def test_the_rail_degree_counts_the_nodes_joined_by_a_rail_edge(self):
        rail_edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])

        assert rail_degree(rail_edge_index, num_nodes=4).tolist() == [2, 1, 1, 0]

    def test_the_hubs_are_the_top_share_of_the_station_zones(self):
        degree = np.array([0, 9, 7, 3, 3, 2, 1, 1, 1, 1, 1])

        classes = zone_classes(degree, hub_share=0.2)

        assert classes.tolist() == ["no station", "hub", "hub"] + ["station"] * 8

    def test_a_tie_across_the_last_hub_leaves_every_zone_of_the_tie_out(self):
        degree = np.array([9, 5, 5, 5, 1, 1, 1, 1, 1, 1])

        classes = zone_classes(degree, hub_share=0.2)

        assert classes.tolist() == ["hub"] + ["station"] * 9


def per_user_ranks() -> pl.DataFrame:
    """Two users and two models. User 0 has a positive at hub node 0 and one at node 2, user 1 one at node 2."""
    return pl.DataFrame({
        "name": ["A", "A", "B", "B"],
        "stage": [PER_USER_STAGE] * 4,
        "seed": [42, 42, 42, 42],
        "user_idx": [0, 1, 0, 1],
        "pos_node": [[0, 2], [2], [0, 2], [2]],
        "pos_hops": [[1.0, 14.0], [3.0], [1.0, 14.0], [3.0]],
        "pos_n_scored_higher": [[0, 99], [9], [3, 299], [9]],
        "pos_n_tied": [[0, 0], [0], [0, 0], [0]],
        "pos_avg_recall": [[1.0, 0.0], [0.8], [0.9, 0.0], [0.8]],
    })


CLASSES = np.array(["hub", "no station", "station"], dtype=object)


class TestRankReads:
    def test_each_positive_carries_its_rank_class_and_hop_band(self):
        positives = positive_frame(per_user_ranks(), CLASSES).filter(name="B", user_idx=0).sort("pos_node")

        assert positives["zone_class"].to_list() == ["hub", "station"]
        assert positives["rank"].to_list() == [4.0, 300.0]
        assert positives["hop_band"].to_list() == ["0-2", "13+"]
        assert positives["n_pos"].to_list() == [2, 2]

    def test_the_zone_split_pairs_the_contribution_of_each_class(self):
        comparison = paired_zone_split(positive_frame(per_user_ranks(), CLASSES), reference="B")

        hub_row = comparison.filter(group="hub")
        assert hub_row["n_users"].item() == 1
        assert hub_row["mean_diff"].item() == pytest.approx((1.0 - 0.9) / 2)

    def test_the_rank_tail_reads_the_positives_beyond_rank_50_of_the_grouping_model(self):
        positives = positive_frame(per_user_ranks(), CLASSES)
        ranks = grouping_ranks(positives, "B")

        tail = paired_rank_tail(positives, "B", ranks)

        assert tail["grouping rank"]["group"].to_list() == ["1-50", "201+"]
        far_row = tail["hop band"].filter(group="13+")
        assert far_row["mean_diff"].item() == pytest.approx(np.log2(100) - np.log2(300))
        assert tail["zone class"]["group"].to_list() == ["station"]


class TestModelReads:
    def test_joined_station_zones_with_equal_embeddings_have_similarity_one(self):
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
        rail_edge_index = torch.tensor([[0, 1], [1, 0]])
        is_station = np.array([True, True, True, False])

        similarity = embedding_similarity(embeddings, rail_edge_index, is_station)

        assert similarity["joined"] == pytest.approx(1.0)
        assert similarity["not_joined"] == pytest.approx(0.0)

    def test_the_probe_records_the_embeddings_and_an_attention_that_sums_to_one(self):
        torch.manual_seed(0)
        network = GAT(1, 2, 1, in_channels=3, hidden_channels=4, out_channels=1, edge_dim=1).eval()
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 3], [1, 0, 2, 1, 3, 0]])
        edge_attr = torch.rand(edge_index.shape[1], 1)
        probe = GATProbe(network)

        with torch.no_grad():
            network(torch.rand(4, 3), edge_index, edge_attr)
        layer_attention = probe.attention()
        probe.remove()

        assert probe.embeddings is not None and probe.embeddings.shape == (4, 4)
        assert len(layer_attention) == 2
        first_edge_index, first_alpha = layer_attention[0]
        classes = np.array(["hub", "station", "station", "station"])
        contiguity_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        summary = attention_summary(first_edge_index, first_alpha, 4, classes, contiguity_edge_index)
        hub_row = summary.filter(zone_class="hub")
        assert hub_row["n_neighbours"].item() == 3
        shares = hub_row.select("self_share", "contiguity_share", "rail_share").row(0)
        assert sum(shares) == pytest.approx(1.0)
        assert shares[2] > 0


class TestRunVariants:
    def test_the_families_are_the_variants_with_both_reduced_arms(self):
        names = [
            "MLP-nonet-nott",
            "GATv2-2-nonet-nott",
            "GATv2-2-nonet-nott-norail",
            "GATv2-2-nonet-nott-nocontig-norail",
            "TwoTower-4",
            "TwoTower-4-norail",
            "Gravity",
        ]

        assert arm_families(names) == ["GATv2-2-nonet-nott"]
        assert grouping_model(names) == "MLP-nonet-nott"

    def test_a_run_without_one_mlp_variant_raises(self):
        with pytest.raises(ValueError, match="exactly one `MLP` variant"):
            grouping_model(["MLP", "MLP-nonet", "GATv2-2"])
