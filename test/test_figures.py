"""Tests for the presentation figure builders (model scheme, node ordering, rank mapping, charts)."""

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import LineString, Point, box

matplotlib.use("Agg")

from activitygraphs.base import CRS
from activitygraphs.figures import (
    DEFAULT_MODELS,
    ROC_MODELS,
    ModelStyle,
    add_transit_line,
    band_axis_labels,
    band_recall_chart,
    edges_within,
    neighbours_of,
    map_figure,
    node_gdf,
    rank_percentile,
    rank_prominence,
    recall_auc_table,
    recall_chart,
    roc_chart,
    score_panels,
    top_k_nodes,
    user_scores,
    with_labels,
    zone_figsize,
    zone_map,
)


@pytest.fixture
def network_nodes() -> gpd.GeoDataFrame:
    loc_ids = ["zone-3", "zone-1", "zone-2"]
    centroids = [Point(i, 0) for i in range(3)]
    polygons = [box(i - 0.4, -0.4, i + 0.4, 0.4) for i in range(3)]

    nodes = gpd.GeoDataFrame(
        {"original_geometry": gpd.GeoSeries(polygons, crs=CRS)},
        geometry=gpd.GeoSeries(centroids, crs=CRS),
    )
    nodes.index = loc_ids
    nodes.index.name = "loc_id"

    return nodes


@pytest.fixture
def curve() -> pl.DataFrame:
    rows = []
    for spec, base in zip(ROC_MODELS, [0.5, 0.45, 0.3, 0.25, 0.2, 0.05], strict=True):
        for k in [1, 10, 100]:
            rows.append({"name": spec.name, "k": k, "mean": base * k / 100, "sd": 0.01})

    return pl.DataFrame(rows)


@pytest.fixture
def band_table() -> pl.DataFrame:
    rows = []
    for spec in DEFAULT_MODELS:
        for band, n_pos, share in [("0-2", 100.0, 0.25), ("2-5", 300.0, 0.75)]:
            rows.append({
                "band": band,
                "name": spec.name,
                "mean": 0.7,
                "sd": 0.01,
                "n_pos": n_pos,
                "share_of_pos": share,
            })

    return pl.DataFrame(rows)


def test_model_scheme_is_consistent():
    names = [spec.name for spec in ROC_MODELS]
    labels = [spec.label for spec in ROC_MODELS]
    colours = [spec.colour for spec in ROC_MODELS]

    assert len(set(names)) == len(set(labels)) == len(set(colours)) == len(ROC_MODELS)
    assert DEFAULT_MODELS[0].label == "GNN - GATSkip-8"
    assert [spec.label for spec in ROC_MODELS if spec.name == "Uniform"] == ["Random"]
    assert all(spec.learned for spec in DEFAULT_MODELS[:3])
    assert not any(spec.learned for spec in DEFAULT_MODELS[3:])


def test_with_labels_renames_and_filters(curve):
    specs = (ModelStyle("Gravity", "Gravity", "#000000", learned=False),)
    labelled = with_labels(curve, specs)

    assert labelled["model"].unique().to_list() == ["Gravity"]
    assert labelled.height == 3

    with pytest.raises(KeyError, match="not in the frame"):
        with_labels(curve, (ModelStyle("Nope", "Nope", "#000000", learned=False),))


def test_node_gdf_is_in_sorted_loc_id_order(network_nodes):
    nodes = node_gdf(network_nodes)

    assert nodes["loc_id"].to_list() == ["zone-1", "zone-2", "zone-3"]
    assert nodes.geometry.geom_type.eq("Polygon").all()
    assert node_gdf(network_nodes, polygons=False).geometry.geom_type.eq("Point").all()


@pytest.mark.parametrize("level_names", [None, ["source", "target"]])
def test_edges_within_keeps_only_interior_edges(network_nodes, level_names):
    pairs = [("zone-1", "zone-2"), ("zone-2", "zone-3"), ("zone-3", "zone-4")]
    edges = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([LineString([(0, 0), (1, 0)])] * 3, crs=CRS)},
        index=pd.MultiIndex.from_tuples(pairs, names=level_names),
    )

    kept = edges_within(edges, node_gdf(network_nodes).iloc[[0, 1]])

    assert list(kept.index) == [("zone-1", "zone-2")]


def test_rank_percentile_puts_the_top_score_at_one():
    percentile = rank_percentile(np.array([0.1, 5.0, -2.0, 0.5]))

    assert percentile[1] == pytest.approx(1.0)
    assert percentile[2] == pytest.approx(0.0)


def test_rank_prominence_separates_the_top_of_the_ranking():
    scores = np.arange(100.0)
    prominence = rank_prominence(scores)

    assert prominence[99] == pytest.approx(1.0)
    assert prominence[0] == pytest.approx(0.0)
    # The gap between ranks 1 and 2 dwarfs the gap between ranks 50 and 51, which a percentile would
    # make identical.
    top_gap = prominence[99] - prominence[98]
    middle_gap = prominence[50] - prominence[49]
    assert top_gap > 5 * middle_gap


def test_top_k_nodes_returns_the_highest_scoring_indices():
    assert top_k_nodes(np.array([0.1, 5.0, -2.0, 0.5]), 2).tolist() == [1, 3]


def test_user_scores_selects_the_model_seed_and_user():
    vectors = pl.DataFrame({
        "name": ["GAT", "GAT", "Gravity"],
        "seed": [0, 1, None],
        "user_id": [1, 1, 1],
        "scores": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    })

    assert user_scores(vectors, "GAT", 1, seed=1).tolist() == [3.0, 4.0]
    assert user_scores(vectors, "Gravity", 1).tolist() == [5.0, 6.0]

    with pytest.raises(KeyError):
        user_scores(vectors, "GAT", 2, seed=0)


def test_zone_map_draws_choropleth_zones_and_highlights(network_nodes):
    nodes = node_gdf(network_nodes)

    ax = zone_map(
        nodes,
        values=np.array([0.1, 0.5, 0.9]),
        highlight_zones={"Home": [0], "Visited": [1]},
        markers={"Work": [2]},
        highlight=[2],
        show_nodes=True,
        basemap=False,
    )

    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Home", "Visited", "Work"]


def test_zone_map_focus_restricts_the_view(network_nodes):
    nodes = node_gdf(network_nodes)

    full = zone_map(nodes, basemap=False)
    focused = zone_map(nodes, focus=[0], basemap=False)

    assert focused.get_xlim()[1] - focused.get_xlim()[0] < full.get_xlim()[1] - full.get_xlim()[0]


def test_zero_focus_margin_crops_exactly_to_the_zones(network_nodes):
    nodes = node_gdf(network_nodes)

    ax = zone_map(nodes, focus=range(len(nodes)), focus_margin=0.0, basemap=False)

    min_x, min_y, max_x, max_y = nodes.total_bounds
    assert ax.get_xlim() == pytest.approx((min_x, max_x))
    assert ax.get_ylim() == pytest.approx((min_y, max_y))


def test_zone_figsize_matches_the_ground_aspect(network_nodes):
    nodes = node_gdf(network_nodes)
    min_x, min_y, max_x, max_y = nodes.to_crs(nodes.estimate_utm_crs()).total_bounds

    width, height = zone_figsize(nodes, width=7.0)

    assert width == 7.0
    assert height / width == pytest.approx((max_y - min_y) / (max_x - min_x))


def test_map_figure_keeps_the_map_area_whatever_keys_it_reserves(network_nodes):
    nodes = node_gdf(network_nodes)
    width, height = zone_figsize(nodes, width=7.0)

    plain = map_figure(nodes, width=7.0)
    keyed = map_figure(nodes, width=7.0, legend_width=1.5, colour_bar_height=1.0)

    for fig, ax, cax in (plain, keyed):
        assert ax.get_position().width * fig.get_figwidth() == pytest.approx(width)
        assert ax.get_position().height * fig.get_figheight() == pytest.approx(height)

    assert plain[2] is None
    assert keyed[0].get_figwidth() - plain[0].get_figwidth() == pytest.approx(1.5)
    assert keyed[0].get_figheight() - plain[0].get_figheight() == pytest.approx(1.0)
    assert keyed[2].get_position().y1 < keyed[1].get_position().y0


def test_neighbours_of_is_symmetric_and_restricted_to_the_given_nodes(network_nodes):
    nodes = node_gdf(network_nodes)
    # zone-2 appears once as a target and twice as a source; zone-9 is outside `nodes`.
    pairs = [("zone-1", "zone-2"), ("zone-3", "zone-2"), ("zone-2", "zone-9")]
    edges = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([LineString([(0, 0), (1, 0)])] * 3, crs=CRS)},
        index=pd.MultiIndex.from_tuples(pairs),
    )

    assert neighbours_of(edges, nodes, 1) == [0, 2]
    assert neighbours_of(edges, nodes, 0) == [1]


def test_add_transit_line_crosses_the_view_without_widening_it(network_nodes):
    nodes = node_gdf(network_nodes)
    ax = zone_map(nodes, basemap=False)
    xlim = ax.get_xlim()

    add_transit_line(ax, nodes, through=[0, 2], label="Subway")

    assert ax.get_xlim() == xlim
    assert "Subway" in [text.get_text() for text in ax.get_legend().get_texts()]

    drawn = ax.collections[-1].get_paths()[0].vertices
    assert drawn[:, 0].min() < xlim[0]
    assert drawn[:, 0].max() > xlim[1]


def test_add_transit_line_needs_two_distinct_zones(network_nodes):
    nodes = node_gdf(network_nodes)
    ax = zone_map(nodes, basemap=False)

    with pytest.raises(ValueError, match="different zones"):
        add_transit_line(ax, nodes, through=[1, 1])


def test_score_panels_shares_the_colour_scale(network_nodes):
    nodes = node_gdf(network_nodes)
    scores = {"GNN": np.array([0.0, 0.5, 1.0]), "MTF-BPR": np.array([0.2, 0.2, 0.4])}

    fig = score_panels(nodes, scores, basemap=False)

    limits = [ax.collections[0].get_clim() for ax in fig.axes[:2]]
    assert limits[0] == limits[1] == (0.0, 1.0)


def test_charts_build_without_error(curve, band_table):
    assert recall_chart(curve).to_dict()
    assert recall_chart(curve, log_k=False, points=False, tick_ks=[1, 100]).to_dict()
    assert roc_chart(curve, n_nodes=100).to_dict()
    assert band_recall_chart(band_table, budget=50).to_dict()


def test_recall_auc_is_half_for_the_diagonal():
    diagonal = pl.DataFrame({
        "name": ["Uniform"] * 3,
        "k": [1, 50, 100],
        "mean": [0.01, 0.5, 1.0],
        "sd": [None] * 3,
    })
    specs = (ModelStyle("Uniform", "Random", "#000000", learned=False),)

    auc = recall_auc_table(diagonal, n_nodes=100, specs=specs)

    assert auc["auc"][0] == pytest.approx(0.5, abs=1e-3)


def test_band_recall_chart_writes_each_band_distance_and_its_positives(band_table):
    rows = band_axis_labels(band_table).unique(subset=["band"], maintain_order=True)

    assert rows["band_label"].to_list() == ["0-2 km", "2-5 km"]
    assert rows["band_positives"].to_list() == ["100 visits (25%)", "300 visits (75%)"]
    assert band_recall_chart(band_table, budget=50).to_dict()
