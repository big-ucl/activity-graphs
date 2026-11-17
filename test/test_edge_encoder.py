"""Unit tests for the edge encoder: the edges of each kind in use, merged into one edge per pair of nodes."""

import types
from dataclasses import replace

import pytest
import torch
from encoders import continuous_schema
from graphs import make_locations

from activitygraphs.datasets import Locations
from activitygraphs.models.features import EdgeEncoder, EdgeSource, InputFeatures, build_edge_encoder
from activitygraphs.schema import EdgeKind
from activitygraphs.tensors import EdgeTensors


def make_path_and_rail_locations() -> Locations:
    """Return the path ``0 - 1 - 2`` with one feature per edge, and rail edges ``0 - 1`` and ``0 - 2`` with two each."""
    contiguity_edges = EdgeTensors(
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), torch.tensor([[10.0], [11.0], [12.0], [13.0]])
    )
    rail_edges = EdgeTensors(
        torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
    )
    locations = make_locations(torch.zeros(3, 0), contiguity_edges.index)

    return replace(locations, edges={EdgeKind.CONTIGUITY: contiguity_edges, EdgeKind.RAIL: rail_edges})


def edge_source(kind: EdgeKind, width: int) -> EdgeSource:
    return EdgeSource(kind, continuous_schema(kind.group, width))


CONTIGUITY_SOURCE = edge_source(EdgeKind.CONTIGUITY, 1)
RAIL_SOURCE = edge_source(EdgeKind.RAIL, 2)


def attr_by_pair(edges: EdgeTensors) -> dict[tuple[int, int], list[float]]:
    pairs = [tuple(pair) for pair in edges.index.T.tolist()]
    return dict(zip(pairs, edges.attr.tolist(), strict=True))


class TestEdgeEncoder:
    @pytest.mark.parametrize("source", [CONTIGUITY_SOURCE, RAIL_SOURCE], ids=str)
    def test_a_single_source_keeps_its_own_edges(self, source):
        locations = make_path_and_rail_locations()

        assert EdgeEncoder([source])(locations) is locations.edges[source.kind]

    def test_several_sources_hold_each_ordered_pair_once_in_order(self):
        merged_edges = EdgeEncoder([CONTIGUITY_SOURCE, RAIL_SOURCE])(make_path_and_rail_locations())

        assert merged_edges.index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]

    def test_a_pair_of_both_sources_carries_both_flags_and_the_features_of_both(self):
        merged_attr = attr_by_pair(EdgeEncoder([CONTIGUITY_SOURCE, RAIL_SOURCE])(make_path_and_rail_locations()))

        assert merged_attr[(0, 1)] == [1.0, 1.0, 10.0, 1.0, 2.0]
        assert merged_attr[(1, 0)] == [1.0, 1.0, 11.0, 1.0, 2.0]

    def test_a_pair_of_one_source_carries_zeros_for_the_other_source(self):
        merged_attr = attr_by_pair(EdgeEncoder([CONTIGUITY_SOURCE, RAIL_SOURCE])(make_path_and_rail_locations()))

        assert merged_attr[(0, 2)] == [0.0, 1.0, 0.0, 3.0, 4.0]
        assert merged_attr[(1, 2)] == [1.0, 0.0, 12.0, 0.0, 0.0]

    def test_no_source_holds_no_edge(self):
        edges = EdgeEncoder([])(make_path_and_rail_locations())

        assert edges.index.shape == (2, 0)
        assert edges.attr.shape == (0, 0)

    @pytest.mark.parametrize(
        "sources", [[], [CONTIGUITY_SOURCE], [RAIL_SOURCE], [CONTIGUITY_SOURCE, RAIL_SOURCE]], ids=str
    )
    def test_the_width_is_the_number_of_features_of_each_edge(self, sources):
        encoder = EdgeEncoder(sources)

        assert encoder(make_path_and_rail_locations()).attr.shape[1] == encoder.out_dim


@pytest.mark.parametrize("use_contiguity_edges", [False, True])
@pytest.mark.parametrize("use_rail_edges", [False, True])
def test_the_flags_pick_the_edge_kinds_in_order(use_contiguity_edges, use_rail_edges):
    features = InputFeatures(use_contiguity_edges=use_contiguity_edges, use_rail_edges=use_rail_edges)
    schema_edges = {EdgeKind.CONTIGUITY: CONTIGUITY_SOURCE.schema, EdgeKind.RAIL: RAIL_SOURCE.schema}
    schema = types.SimpleNamespace(edges=schema_edges)

    encoder = build_edge_encoder(features, schema)  # type: ignore[arg-type]

    expected_kinds = [EdgeKind.CONTIGUITY] * use_contiguity_edges + [EdgeKind.RAIL] * use_rail_edges
    assert [source.kind for source in encoder.sources] == expected_kinds
