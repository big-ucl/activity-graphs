"""Build the hEART presentation figures into ``reports/figures/heart/``.

Run with ``uv run python notebooks/heart/figures.py``. Every figure is read off one run of the CMAP
dataset, pinned by ``cfg.analysis.run``, so nothing here re-trains a model.

The marimo notebook beside this file, ``figures_nb.py``, imports these builders and drives them from
its widgets, so a panel is defined once and looks the same whichever way it is produced. The
imaginary transport network of figure 1 is added in PowerPoint on top of the exported panels.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from activitygraphs import analysis, figures
from activitygraphs.config import load_config
from activitygraphs.ml.training import PER_USER_STAGE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "cmap"

# The model, seed and zone counts the exported panels are drawn for; the notebook overrides them per widget.
MAP_MODEL = "GATSkip-8-res-hpe-dist"
MAP_SEED = 42
SPATIAL_MODEL = "GATSkip-8-res-hpe-dist"
FLAT_MODEL = "MLP-dist"
MF_MODEL = "HomeZoneMF"
ZONES_IN_VIEW = 30
TOP_KS = (5, 10, 20, 50)

# Figure 1 renders this many times the viewed zones, so the cropped frame has no bare corners.
RENDER_RING = 4

# Every map panel is built at this geometry, so the frames of one animation can be laid over each other.
PANEL_WIDTH = 7.0
LEGEND_WIDTH = 1.7
COLOUR_BAR_HEIGHT = 1.0

# Figure 1's zones are a flat fill rather than a choropleth, so the basemap under them can read through.
ZONE_ALPHA = 0.2

# The tallest the figure 2 and 3 panels may be relative to their width.
PANEL_MAX_ASPECT = 1.4

RECALL_TICK_KS = (1, 10, 20, 30, 40, 50)


# =========================================
# Loading
# =========================================


def load_run():
    """The config, per-user results, score vectors, home distances and node count of the pinned run."""
    cfg = load_config(PROJECT_ROOT, verbose=False, data=DATASET)
    reports = PROJECT_ROOT / cfg.paths.reports_data

    _, per_user = analysis.load_report_run(reports, cfg.data.name, cfg.analysis.run, cfg.train.max_recall_k)
    score_vectors = analysis.load_score_vectors(reports, cfg.data.name, cfg.analysis.run)
    home_distances = analysis.load_home_distances(reports, cfg.data.name, cfg.analysis.run)
    n_nodes = len(score_vectors["scores"].to_list()[0])

    return cfg, per_user, score_vectors, home_distances, n_nodes


def load_network(cfg):
    """The region's zones in PyG node order and its adjacency edges, read from the processed cache."""
    network_path = PROJECT_ROOT / cfg.data.paths.processed / "NetworkGraph"
    nodes = figures.node_gdf(gpd.read_parquet(network_path / "nodes.parquet"))

    return nodes, gpd.read_parquet(network_path / "edges.parquet")


def out_dir(cfg) -> Path:
    """The directory the panels are written to, created if it is not there."""
    path = PROJECT_ROOT / cfg.paths.figures / "heart"
    path.mkdir(parents=True, exist_ok=True)

    return path


def save(fig_or_ax, path: Path, name: str, dpi: int = 300, tight: bool = True):
    """Write a matplotlib figure or axes to ``<name>.png`` on a white background.

    Args:
        fig_or_ax: The figure to write, or an axes of it.
        path: Directory to write into.
        name: File stem.
        dpi: Raster resolution.
        tight: Crop to the drawn content. False writes the figure at its own size, which is what keeps
            the panels built by ``figures.map_figure`` identical.

    Returns:
        The figure written.
    """
    fig = fig_or_ax.figure
    fig.savefig(path / f"{name}.png", dpi=dpi, bbox_inches="tight" if tight else None, facecolor="white")
    print(f"  wrote {name}.png")

    return fig


def save_chart(chart, path: Path, name: str, scale_factor: int = 3):
    """Write an Altair chart to ``<name>.png``."""
    chart.save(path / f"{name}.png", scale_factor=scale_factor)
    print(f"  wrote {name}.png")

    return chart


# =========================================
# The individual the maps are drawn for
# =========================================


def users_by_visits(per_user: pl.DataFrame) -> list[int]:
    """Test users ordered by how many zones they visited, most first.

    Users with several visited zones are the ones that make figures 2c and 2e worth looking at.
    """
    rows = per_user.filter(pl.col("stage") == PER_USER_STAGE, pl.col("name") == "Gravity")

    return (
        rows
        .with_columns(n_visited=pl.col("pos_node").list.len())
        .sort("n_visited", descending=True)["user_id"]
        .to_list()
    )


def user_zones(per_user: pl.DataFrame, user_id: int) -> tuple[int, list[int]]:
    """The home node and the visited nodes of one test user."""
    row = per_user.filter(pl.col("stage") == PER_USER_STAGE, pl.col("user_id") == user_id).row(0, named=True)

    return int(row["home_node"]), [int(node) for node in row["pos_node"]]


# =========================================
# Figure 1 - graph structure
# =========================================


def neighbourhood_of(nodes: gpd.GeoDataFrame, home_node: int, zones_in_view: int = ZONES_IN_VIEW):
    """The zones around a home zone, the map from node index into them, and the ones the view crops to.

    The nearest zones sit in a disc around home, whose bounding box only touches the frame at four
    points. Rendering a wider ring and cropping to the inner one fills the panel with zones.

    Args:
        nodes: Zones in PyG node order from ``figures.node_gdf``.
        home_node: Node index of the home zone.
        zones_in_view: Zones the cropped view holds.

    Returns:
        ``(neighbourhood, local, view)``: the wider ring as its own frame, a node index -> row map into
        it, and the rows the view crops to.
    """
    metric_nodes = nodes.to_crs(nodes.estimate_utm_crs())
    centre = metric_nodes.geometry.iloc[home_node].centroid
    order = np.argsort(metric_nodes.geometry.centroid.distance(centre).to_numpy())

    view_index = order[:zones_in_view]
    neighbourhood_index = np.sort(order[: zones_in_view * RENDER_RING])

    neighbourhood = nodes.iloc[neighbourhood_index].reset_index(drop=True)
    local = {int(node): i for i, node in enumerate(neighbourhood_index)}

    return neighbourhood, local, [local[int(node)] for node in view_index]


def activity_markers(edges: gpd.GeoDataFrame, neighbourhood: gpd.GeoDataFrame, local: dict, view: list, home_node: int):
    """Home, work and gym markers for figure 1, and the zones the subway line runs through.

    Two work zones are picked from opposite ends of the view, then a gym on a zone adjacent to each.
    """
    within = figures.edges_within(edges, neighbourhood)
    home = local[home_node]

    work = [view[len(view) // 3], view[-2]]
    visible = set(view)
    gyms = [
        next(n for n in figures.neighbours_of(within, neighbourhood, w) if n in visible and n not in {home, *work})
        for w in work
    ]

    return {"Home": [home], "Work": work, "Gym": gyms}, {"Home": [home]}, [home, gyms[0]]


def graph_structure_figures(nodes, edges, home_node: int, path: Path, zones_in_view: int = ZONES_IN_VIEW) -> None:
    """Figure 1: (a) plain zones, (b) activity markers, (c) the adjacency graph, (d) a subway line, (e) home
    alone on the graph, (f) the graph over the zones with nothing marked.

    Every panel reserves the same strip for the legend, so the panels with one and the panels without it
    still draw the zones at the same size and in the same place.
    """
    neighbourhood, local, view = neighbourhood_of(nodes, home_node, zones_in_view)
    markers, home_marker, subway_through = activity_markers(edges, neighbourhood, local, view, home_node)
    within = figures.edges_within(edges, neighbourhood)
    graph = {"edges": within, "show_nodes": True}

    def panel(**kwargs):
        _, ax, _ = figures.map_figure(neighbourhood.iloc[view], width=PANEL_WIDTH, legend_width=LEGEND_WIDTH)

        return figures.zone_map(neighbourhood, ax=ax, focus=view, focus_margin=0.0, alpha=ZONE_ALPHA, **kwargs)

    save(panel(), path, "fig1a-zones", tight=False)
    save(panel(markers=markers), path, "fig1b-activities", tight=False)
    save(panel(markers=markers, **graph), path, "fig1c-adjacency", tight=False)
    save(panel(markers=home_marker, **graph), path, "fig1e-home-graph", tight=False)
    save(panel(**graph), path, "fig1f-graph", tight=False)

    transit_ax = panel(markers=markers, **graph)
    figures.add_transit_line(transit_ax, neighbourhood, through=subway_through, label="Subway")
    save(transit_ax, path, "fig1d-transit", tight=False)


# =========================================
# Figure 2 - what a ranking looks like
# =========================================


def ranking_focus(home_node: int, scores: list[np.ndarray], k: int = max(TOP_KS)) -> list[int]:
    """The zones a ranking map crops to: the home zone and each model's top ``k``.

    Figures 2 and 3 share this view, so the selections of figure 2 (e) can be read against the zones of the
    panels the animation builds up to them. A visited zone at the far end of the region is left out of it: it
    would stretch the panel over tens of kilometres of zones no model ranked.

    Args:
        home_node: Node index of the user's home zone.
        scores: One score vector per model whose selection should stay in the view.
        k: Size of the selection kept in the view.

    Returns:
        Node indices, ascending.
    """
    selected = {int(node) for vector in scores for node in figures.top_k_nodes(vector, k)}

    return sorted({home_node, *selected})


def crop(focus: list[int], home_node: int) -> dict:
    """The ``figures.zone_map`` arguments that put a figure 2 or 3 panel on the shared view.

    The view is flattened to ``PANEL_MAX_ASPECT`` around the home zone, which keeps the home zone and the
    selection around it in frame and lets a visited zone at the far end of the region fall outside it.
    """
    return {"focus": focus, "max_aspect": PANEL_MAX_ASPECT, "aspect_anchor": home_node}


def map_panel(nodes, focus: list[int], home_node: int, colour_bar: bool = False):
    """A figure-2 panel: the region cropped to ``focus``, at the geometry every panel of the figure shares.

    A panel with no choropleth keeps the strip the colour bar occupies, so it stays the same size as the
    panels that have one, but draws nothing in it.
    """
    fig, ax, cax = figures.map_figure(
        nodes.iloc[focus],
        width=PANEL_WIDTH,
        legend_width=LEGEND_WIDTH,
        colour_bar_height=COLOUR_BAR_HEIGHT,
        max_aspect=PANEL_MAX_ASPECT,
    )

    if colour_bar:
        return fig, ax, cax

    cax.set_axis_off()

    return fig, ax, None


def region_figures(
    nodes,
    score_vectors: pl.DataFrame,
    user_id: int,
    home_node: int,
    visited_nodes: list[int],
    path: Path,
    model: str = MAP_MODEL,
    seed: int | None = MAP_SEED,
) -> None:
    """Figure 2 (a) to (c): the region plain, with the home zone, and with the visited zones.

    The zones are filled in the colour the choropleth of (d) and (e) gives a zone outside its top ranks, and
    the view is the one those panels crop to, so the animation only ever adds to the frame before it.
    """
    focus = ranking_focus(home_node, [figures.user_scores(score_vectors, model, user_id, seed)])

    def panel(**kwargs):
        _, ax, _ = map_panel(nodes, focus, home_node)

        return figures.zone_map(
            nodes, ax=ax, fill=figures.MAP_BELOW_FLOOR, boundary_width=0.3, **crop(focus, home_node), **kwargs
        )

    save(panel(), path, "fig2a-region", tight=False)
    save(panel(highlight_zones={"Home": [home_node]}), path, "fig2b-home", tight=False)
    save(
        panel(highlight_zones={"Home": [home_node], "Visited": visited_nodes}),
        path,
        "fig2c-visited",
        tight=False,
    )


def ranking_figures(
    nodes,
    score_vectors: pl.DataFrame,
    user_id: int,
    home_node: int,
    visited_nodes: list[int],
    n_nodes: int,
    path: Path,
    model: str = MAP_MODEL,
    seed: int | None = MAP_SEED,
    ks: tuple[int, ...] = TOP_KS,
) -> None:
    """Figure 2 (d) and (e): the score choropleth, and the top-k selections, each cutoff on its own panel
    and the whole set side by side.

    The choropleth is coloured by ``figures.rank_prominence`` rather than a percentile, which would spread
    colour evenly over the region and flatten the top of the ranking. The scale is floored at
    ``figures.MAP_RANK_FLOOR``, so zones ranked below it are drawn white and the whole ramp is spent on the
    ranks a candidate set is drawn from.
    """
    scores = figures.user_scores(score_vectors, model, user_id, seed)
    prominence = figures.rank_prominence(scores)
    focus = ranking_focus(home_node, [scores], k=max(ks))
    choropleth = {
        "values": prominence,
        "legend": False,
        "boundary_width": 0.3,
        **crop(focus, home_node),
        **figures.rank_limits(n_nodes),
    }

    fig, ax, cax = map_panel(nodes, focus, home_node, colour_bar=True)
    figures.zone_map(nodes, ax=ax, highlight_zones={"Home": [home_node]}, **choropleth)
    figures.rank_colour_bar(fig, ax, n_nodes, cax=cax)
    save(fig, path, "fig2d-scores", tight=False)

    for k in ks:
        fig, ax, cax = map_panel(nodes, focus, home_node, colour_bar=True)
        figures.zone_map(
            nodes,
            ax=ax,
            highlight_zones={"Home": [home_node], "Visited": visited_nodes},
            highlight=figures.top_k_nodes(scores, k),
            **choropleth,
        )
        figures.rank_colour_bar(fig, ax, n_nodes, cax=cax)
        save(fig, path, f"fig2e-topk-{k}", tight=False)

    fig, axes = plt.subplots(1, len(ks), figsize=(12, 7), sharex=True, sharey=True)

    for panel, k in zip(axes, ks, strict=True):
        figures.zone_map(
            nodes,
            ax=panel,
            highlight_zones={"Home": [home_node], "Visited": visited_nodes},
            highlight=figures.top_k_nodes(scores, k),
            title=f"top {k}",
            **choropleth,
        )

    figures.rank_colour_bar(fig, list(axes), n_nodes)
    save(fig, path, "fig2e-topk")


# =========================================
# Figure 3 - spatial vs non-spatial ranking
# =========================================


def model_panels(nodes, scores: dict[str, np.ndarray], home_node: int, visited_nodes: list[int], n_nodes: int, focus):
    """One rank choropleth per model, side by side over a shared view and under one colour bar.

    Args:
        nodes: Zones in PyG node order from ``figures.node_gdf``.
        scores: Model label -> the user's score vector under it, in the order the panels are drawn.
        home_node: Node index of the user's home zone.
        visited_nodes: Node indices the user visited.
        n_nodes: Zones in the region, which the colour scale is read against.
        focus: Node indices the view crops to.

    Returns:
        The figure holding the panels.
    """
    map_width, map_height = figures.zone_figsize(nodes.iloc[focus], width=PANEL_WIDTH, max_aspect=PANEL_MAX_ASPECT)
    fig, axes = plt.subplots(
        1,
        len(scores),
        figsize=(len(scores) * map_width, map_height + COLOUR_BAR_HEIGHT),
        gridspec_kw={"wspace": 0.03},
    )

    for ax, (label, vector) in zip(axes, scores.items(), strict=True):
        figures.zone_map(
            nodes,
            values=figures.rank_prominence(vector),
            highlight_zones={"Home": [home_node], "Visited": visited_nodes},
            ax=ax,
            legend=False,
            boundary_width=0.3,
            title=label,
            **crop(focus, home_node),
            **figures.rank_limits(n_nodes),
        )

    figures.rank_colour_bar(fig, list(axes), n_nodes)

    return fig


def spatial_vs_flat_figure(
    nodes,
    score_vectors: pl.DataFrame,
    user_id: int,
    home_node: int,
    visited_nodes: list[int],
    n_nodes: int,
    path: Path,
    spatial: str = SPATIAL_MODEL,
    flat: str = FLAT_MODEL,
    factorization: str = MF_MODEL,
    seed: int | None = MAP_SEED,
) -> None:
    """Figure 3: the same user under a GNN, a non-spatial model and the factorization baseline.

    (a) holds the first two and (b) adds the third. Both crop to the view of figure 2 (e), widened to hold
    whichever zones the two ranking models select, and share one colour bar, so a panel drawn in both is the
    same panel. The factorization baseline ranks by home-zone popularity and picks zones across the whole
    region, so it is not given the view: it would pull the frame out to the region's own bounds.
    """
    labels = {spec.name: spec.label for spec in figures.DEFAULT_MODELS}
    names = [spatial, flat, factorization]
    scores = {labels[name]: figures.user_scores(score_vectors, name, user_id, seed) for name in names}
    focus = ranking_focus(home_node, [scores[labels[name]] for name in (spatial, flat)])
    panels = {"fig3a-spatial-vs-flat": len(names) - 1, "fig3b-with-factorization": len(names)}

    for stem, shown in panels.items():
        drawn = dict(list(scores.items())[:shown])
        save(model_panels(nodes, drawn, home_node, visited_nodes, n_nodes, focus), path, stem)


# =========================================
# Figure 4 - recall@k curve
# =========================================


def recall_figures(cfg, per_user: pl.DataFrame, n_nodes: int, path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Figure 4: recall@k at the reported cutoffs, and over the full range against the share of the region.

    Returns:
        ``(auc_table, avg_recall_table)``, the area under each model's curve and its ``avg_recall@K``.
    """
    budget = cfg.train.max_recall_k
    curve = analysis.recall_curve(per_user, list(range(1, budget + 1)))
    save_chart(
        figures.recall_chart(curve, tick_ks=RECALL_TICK_KS, points=False, log_k=False),
        path,
        "fig4a-recall-at-k",
    )

    dense_ks = np.unique(np.geomspace(1, n_nodes, 80).astype(int)).tolist()
    dense_curve = analysis.recall_curve(per_user, dense_ks)
    save_chart(figures.roc_chart(dense_curve, n_nodes), path, "fig4b-recall-roc")

    return (
        figures.recall_auc_table(dense_curve, n_nodes),
        figures.avg_recall_table(per_user, cfg.train.max_recall_k),
    )


# =========================================
# Figure 5 - performance by distance from home
# =========================================


def band_figures(cfg, per_user, home_distances, path: Path) -> pl.DataFrame:
    """Figure 5: mean recall of the visited locations in each distance band from home.

    Returns:
        The distance-band recall table behind the chart.
    """
    keep = [spec.name for spec in figures.DEFAULT_MODELS]
    band = analysis.distance_band_recall(
        per_user.filter(pl.col("name").is_in(keep)), home_distances, list(cfg.data.distance_bands)
    )
    band_table = analysis.distance_band_recall_table(band)

    save_chart(figures.band_recall_chart(band_table, cfg.train.max_recall_k), path, "fig5-distance-bands")

    return band_table


def main() -> None:
    cfg, per_user, score_vectors, home_distances, n_nodes = load_run()
    nodes, edges = load_network(cfg)
    path = out_dir(cfg)

    user_id = users_by_visits(per_user)[0]
    home_node, visited_nodes = user_zones(per_user, user_id)
    print(f"user {user_id}: home node {home_node}, {len(visited_nodes)} visited zones, {n_nodes} zones in the region")

    print("figure 1 - graph structure:")
    graph_structure_figures(nodes, edges, home_node, path)
    print("figure 2 - what a ranking looks like:")
    region_figures(nodes, score_vectors, user_id, home_node, visited_nodes, path)
    ranking_figures(nodes, score_vectors, user_id, home_node, visited_nodes, n_nodes, path)
    print("figure 3 - spatial vs non-spatial:")
    spatial_vs_flat_figure(nodes, score_vectors, user_id, home_node, visited_nodes, n_nodes, path)
    print("figure 4 - recall@k curve:")
    recall_figures(cfg, per_user, n_nodes, path)
    print("figure 5 - performance by distance from home:")
    band_figures(cfg, per_user, home_distances, path)


if __name__ == "__main__":
    main()
