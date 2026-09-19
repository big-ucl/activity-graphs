"""Builders for the presentation figures: matplotlib zone maps and Altair model curves.

Node order is the PyG node order: ``network_nodes`` sorted by ``loc_id``, so row ``i`` of a gdf from
``node_gdf`` is node index ``i`` of a score vector, a ``pos_node`` or a ``home_node``.

``DEFAULT_MODELS`` is the shared scheme behind every chart: the models of the comparison, the names
they are shown under, and their colours - saturated for the learned models, muted for the baselines.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import altair as alt
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from shapely.geometry import LineString

from activitygraphs.analysis import AVG_RECALL_COLUMN, per_user_metric_summary
from activitygraphs.base import CRS

# =========================================
# Model scheme
# =========================================


SOLID: tuple[int, ...] = (1, 0)
DASHED: tuple[int, ...] = (6, 3)
DOTTED: tuple[int, ...] = (2, 2)


@dataclass(frozen=True)
class ModelStyle:
    """A model of the comparison, the name it is shown under, and how it is drawn.

    Attributes:
        name: Model name as it appears in the results parquets.
        label: Name shown in legends and tables.
        colour: Line and mark colour.
        learned: Whether the model is fitted per training seed, which is what earns a saturated colour.
        dash: Stroke dash pattern as a Vega-Lite ``strokeDash`` range entry, ``SOLID`` for an unbroken line.
    """

    name: str
    label: str
    colour: str
    learned: bool
    dash: tuple[int, ...] = SOLID


DEFAULT_MODELS: tuple[ModelStyle, ...] = (
    ModelStyle("GATSkip-8-res-hpe-dist", "GNN - GATSkip-8", "#d1242f", learned=True),
    ModelStyle("MLP-dist", "MLP-dist", "#1f6feb", learned=True),
    ModelStyle("HomeZoneMF", "MTF-BPR", "#2da44e", learned=True),
    ModelStyle("Gravity", "Gravity", "#8c7b6b", learned=False),
    ModelStyle("VisitFrequency", "VisitFrequency", "#9aa0a6", learned=False),
)
"""The comparison set. ``Uniform`` is left out: it is a flat reference that only earns a line on the
ROC curve, where ``ROC_MODELS`` adds it back as "Random"."""

RANDOM_MODEL = ModelStyle("Uniform", "Random", "#c9ced4", learned=False)

ROC_MODELS: tuple[ModelStyle, ...] = (*DEFAULT_MODELS, RANDOM_MODEL)


def model_labels(specs: Sequence[ModelStyle] = DEFAULT_MODELS) -> list[str]:
    """The display labels of ``specs``, in their scheme order."""
    return [spec.label for spec in specs]


def colour_scale(specs: Sequence[ModelStyle] = DEFAULT_MODELS) -> alt.Scale:
    """An Altair colour scale mapping each spec's label to its colour."""
    return alt.Scale(domain=model_labels(specs), range=[spec.colour for spec in specs])


def model_colour(specs: Sequence[ModelStyle] = DEFAULT_MODELS) -> alt.Color:
    """The shared colour encoding: each spec's label in its colour, with an opaque legend swatch.

    The swatch is drawn as a stroke at full opacity because a layered chart otherwise takes the
    symbol from whichever layer it likes, and a translucent sd band bleaches it.
    """
    return alt.Color(
        "model:N",
        scale=colour_scale(specs),
        sort=model_labels(specs),
        legend=alt.Legend(title="Model", symbolType="stroke", symbolStrokeWidth=3, symbolOpacity=1.0),
    )


def model_dash(specs: Sequence[ModelStyle] = DEFAULT_MODELS) -> alt.StrokeDash:
    """The shared stroke encoding: each spec's label in its dash pattern.

    It carries the same field and title as ``model_colour``, so Vega-Lite merges the two into one
    legend whose swatches show both the colour and the dash.
    """
    return alt.StrokeDash(
        "model:N",
        scale=alt.Scale(domain=model_labels(specs), range=[list(spec.dash) for spec in specs]),
        sort=model_labels(specs),
        legend=alt.Legend(title="Model"),
    )


def with_labels(frame: pl.DataFrame, specs: Sequence[ModelStyle] = DEFAULT_MODELS) -> pl.DataFrame:
    """Restrict a results frame to ``specs`` and add the ``model`` label column they are shown under.

    Args:
        frame: Any frame carrying a ``name`` column, e.g. ``recall_curve`` or ``distance_band_recall_table``.
        specs: The models to keep, in scheme order.

    Returns:
        The matching rows with a ``model`` column added.

    Raises:
        KeyError: If a spec's model is missing from the frame.
    """
    present = set(frame["name"].unique())
    missing = [spec.name for spec in specs if spec.name not in present]

    if missing:
        raise KeyError(f"models {missing} are not in the frame")

    labels = {spec.name: spec.label for spec in specs}

    return frame.filter(pl.col("name").is_in(list(labels))).with_columns(model=pl.col("name").replace_strict(labels))


# =========================================
# Nodes and scores
# =========================================


def node_gdf(network_nodes: gpd.GeoDataFrame, polygons: bool = True) -> gpd.GeoDataFrame:
    """Network nodes in PyG node order, in ``CRS``, with polygon or point geometry.

    Args:
        network_nodes: Nodes from ``load_*_network_graph``, indexed by ``loc_id``.
        polygons: Use the zone polygons (``original_geometry``) rather than the centroids.

    Returns:
        The nodes sorted by ``loc_id`` with a reset index, so row ``i`` is node index ``i``.
    """
    nodes = network_nodes.sort_index().reset_index()
    nodes = nodes.set_geometry("original_geometry") if polygons else nodes.set_geometry("geometry")

    return nodes.to_crs(CRS)


def edges_within(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The adjacency edges with both endpoints among ``nodes``, for drawing a subset of the region.

    The endpoints are read off the two index levels by position: ``contiguity_graph`` leaves them
    unnamed, and only the Geneva builder names them while joining its islands.

    Args:
        edges: Edges from ``load_*_network_graph``, indexed by (source ``loc_id``, target ``loc_id``).
        nodes: The subset of zones to keep edges within, carrying a ``loc_id`` column.

    Returns:
        The rows of ``edges`` whose source and target are both in ``nodes``.
    """
    loc_ids = set(nodes["loc_id"])
    source = edges.index.get_level_values(0)
    target = edges.index.get_level_values(1)

    return edges[source.isin(loc_ids) & target.isin(loc_ids)]


def neighbours_of(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, index: int) -> list[int]:
    """Node indices sharing a contiguity edge with ``nodes.iloc[index]``.

    The graph is undirected but stored with one row per ordered pair, so both index levels are
    searched.

    Args:
        edges: Edges from ``load_*_network_graph``, indexed by (source ``loc_id``, target ``loc_id``).
        nodes: The zones ``index`` and the result are positions into, carrying a ``loc_id`` column.
        index: Position of the zone whose neighbours are wanted.

    Returns:
        The positions into ``nodes`` of its neighbours that are themselves in ``nodes``, ascending.
    """
    loc_id = nodes["loc_id"].iloc[index]
    source = edges.index.get_level_values(0)
    target = edges.index.get_level_values(1)

    adjacent = set(target[source == loc_id]) | set(source[target == loc_id])
    position = {value: i for i, value in enumerate(nodes["loc_id"])}

    return sorted(position[value] for value in adjacent if value in position)


def user_scores(score_vectors: pl.DataFrame, name: str, user_id: int, seed: int | None = None) -> np.ndarray:
    """One user's score vector for one model, as an array over nodes in PyG node order.

    Args:
        score_vectors: Score vectors from ``load_score_vectors``.
        name: Model name.
        user_id: Test user to pull.
        seed: Training seed; ``None`` selects the seedless baseline rows.

    Returns:
        Float array of shape ``[n_nodes]``.

    Raises:
        KeyError: If the model, seed and user select no row.
    """
    rows = score_vectors.filter(
        pl.col("name") == name,
        pl.col("user_id") == user_id,
        pl.col("seed").is_null() if seed is None else pl.col("seed") == seed,
    )

    if rows.is_empty():
        raise KeyError(f"no score vector for model {name!r}, user {user_id!r}, seed {seed!r}")

    return np.asarray(rows["scores"].to_list()[0], dtype=np.float64)


def rank_percentile(scores: np.ndarray) -> np.ndarray:
    """Scores mapped to their rank percentile, 1.0 being the top-ranked node.

    Models on different score scales become comparable on a shared colour bar.
    """
    order = np.argsort(np.argsort(scores))

    return order / (len(scores) - 1)


def rank_prominence(scores: np.ndarray) -> np.ndarray:
    """Scores mapped to ``1 - log(rank) / log(n)``, 1.0 being the top-ranked node.

    A percentile spreads colour evenly over the region by construction, which flattens exactly the
    top of the ranking a choropleth is meant to show; on a log rank the top few zones separate and
    the long tail of unranked ones stays uniformly dark.
    """
    ranks = np.argsort(np.argsort(-scores)) + 1

    return 1.0 - np.log(ranks) / np.log(len(scores))


def prominence_of_rank(rank: int, n_nodes: int) -> float:
    """The ``rank_prominence`` value a zone of this rank takes, the inverse of that mapping.

    Args:
        rank: Rank of the zone, 1 being the highest scored.
        n_nodes: Zones in the region.

    Returns:
        The prominence of that rank, 1.0 at rank 1.
    """
    return 1.0 - np.log(rank) / np.log(n_nodes)


def top_k_nodes(scores: np.ndarray, k: int) -> np.ndarray:
    """Node indices of the ``k`` highest-scoring nodes."""
    return np.argsort(-scores, kind="stable")[:k]


# =========================================
# Zone maps (matplotlib)
# =========================================

BASEMAP = cx.providers.Esri.WorldGrayCanvas
"""Carto's CDN serves an "API KEY REQUIRED" watermark over its Positron tiles; Esri's grey canvas is
the equivalent unkeyed light basemap."""

MARKER_COLOURS = ["#9f1853", "#1192e8", "#005d5d", "#8a3ffc", "#fa4d56", "#570408"]
ZONE_COLOURS = ["#C83E4D", "#F2AB40", "#005d5d", "#8a3ffc"]
EDGE_COLOUR = "#1f6feb"
NODE_COLOUR = "#1f6feb"
TRANSIT_COLOUR = "#f59f00"
BOUNDARY_COLOUR = "#71717a"
HIGHLIGHT_COLOUR = "#000000"
ZONE_FILL = "#c5d8f0"

MAP_LEGEND = {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0), "borderaxespad": 0.0, "frameon": False}
"""The zones fill the frame, so a legend inside the axes always covers some of them."""

MAP_BELOW_FLOOR = "#ffffff"
MAP_CMAP = LinearSegmentedColormap.from_list(
    "blues-mid", plt.get_cmap("Blues")(np.linspace(0.00, 0.92, 256))
).with_extremes(under=MAP_BELOW_FLOOR)
"""A single-hue ramp, which leaves the rest of the colour wheel free for the markers drawn over it. Values
below the scale's floor take ``MAP_BELOW_FLOOR``, so the floor reads as a boundary rather than a gradient."""

MAP_RANK_FLOOR = 500
MAP_RANK_TICKS = (1, 5, 20, 50, 150, 300, 500)
"""Rank prominence is already logarithmic in rank, so a ramp stretched over it is log-spaced in rank whatever
its limits. What buys resolution among the top-ranked zones is the floor: everything ranked below it is drawn
in one flat colour, leaving the whole ramp for the ranks a candidate set is drawn from."""


def rank_limits(n_nodes: int, floor: int = MAP_RANK_FLOOR) -> dict[str, float]:
    """The ``vmin``/``vmax`` that spend a ``rank_prominence`` colour scale on the top ``floor`` ranks.

    Args:
        n_nodes: Zones in the region.
        floor: Worst rank the ramp still colours; everything below it takes ``MAP_BELOW_FLOOR``.

    Returns:
        A mapping to splat into ``zone_map`` or ``score_panels``.
    """
    return {"vmin": prominence_of_rank(floor, n_nodes), "vmax": 1.0}


def rank_colour_bar(
    fig: Figure,
    axes,
    n_nodes: int,
    floor: int = MAP_RANK_FLOOR,
    ticks: Sequence[int] = MAP_RANK_TICKS,
    label: str | None = None,
    **kwargs,
):
    """A colour bar for a ``rank_prominence`` choropleth, ticked in ranks rather than in prominence.

    Prominence is not a quantity a reader of the map is after, so the bar is labelled by the rank each
    step stands for. It is built from ``MAP_CMAP`` and ``floor``, which the panels must be drawn with too.

    Args:
        fig: Figure the bar is added to.
        axes: Axes the bar is stolen from, one or a list.
        n_nodes: Zones in the region.
        floor: Worst rank the ramp still colours, matching the panels' ``rank_limits``.
        ticks: Ranks to label.
        label: Bar label; a default naming the floor when None.
        **kwargs: Forwarded to ``Figure.colorbar``.

    Returns:
        The colour bar.
    """
    from matplotlib.cm import ScalarMappable

    settings = {"orientation": "horizontal", "fraction": 0.035, "pad": 0.02, "shrink": 0.55, "extend": "min"}
    settings.update(kwargs)

    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(prominence_of_rank(floor, n_nodes), 1.0), cmap=MAP_CMAP), ax=axes, **settings
    )
    bar.set_ticks([prominence_of_rank(rank, n_nodes) for rank in ticks])
    bar.set_ticklabels([str(rank) for rank in ticks])
    bar.set_label(
        label or f"rank of zone in the model's ranking (white: outside the top {floor})", fontsize=10
    )
    bar.ax.tick_params(labelsize=9)

    return bar


def zone_map(
    nodes: gpd.GeoDataFrame,
    values: np.ndarray | None = None,
    markers: dict[str, Sequence[int]] | None = None,
    highlight_zones: dict[str, Sequence[int]] | None = None,
    edges: gpd.GeoDataFrame | None = None,
    show_nodes: bool = False,
    highlight: Sequence[int] | None = None,
    focus: Sequence[int] | None = None,
    focus_margin: float = 0.25,
    max_aspect: float | None = None,
    aspect_anchor: int | None = None,
    ax: Axes | None = None,
    fill: str = ZONE_FILL,
    alpha: float = 0.85,
    cmap: str | Colormap = MAP_CMAP,
    zone_colours: Sequence[str] = ZONE_COLOURS,
    zone_edge: str = "none",
    zone_edge_width: float = 1.2,
    highlight_colour: str = HIGHLIGHT_COLOUR,
    highlight_width: float = 0.8,
    vmin: float | None = None,
    vmax: float | None = None,
    legend: bool = True,
    basemap: bool = True,
    boundary_width: float = 0.5,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
) -> Axes:
    """Draw the zones, optionally as a choropleth with marked zones and an adjacency overlay.

    Args:
        nodes: Zones in PyG node order from ``node_gdf``, already subset to what should be drawn.
        values: Value per node to colour by, of shape ``[len(nodes)]``; a flat fill if None.
        markers: Label -> node indices (into ``nodes``) drawn as labelled centroid points.
        highlight_zones: Label -> node indices filled in a flat colour, drawn over the choropleth.
        edges: Adjacency edges from ``load_*_network_graph`` drawn as an overlay.
        show_nodes: Draw a dot at each zone's centroid, so the adjacency overlay reads as a graph.
        highlight: Node indices (into ``nodes``) outlined in bold, e.g. a top-k selection.
        focus: Node indices whose bounds set the view, so a selection clustered in one corner of the
            region fills the panel while the rest of the region is still drawn around it.
        focus_margin: Padding around ``focus``, as a fraction of its own width and height. 0 crops
            the view to the zones themselves, with ``zone_figsize`` to keep them from letterboxing.
        max_aspect: Largest ground height over width the view may take. A taller view keeps its width and
            gives up height, which is what stops one far-off zone stretching the panel into a column;
            ``zone_figsize`` takes it too, so the figure keeps matching the view.
        aspect_anchor: Node index the height given up is taken around, its own zone staying in view; the
            focus's own centre when None.
        ax: Axes to draw on; a new figure is made if None.
        fill: Flat colour the zones are filled with when ``values`` is None.
        alpha: Opacity of the zone fill, flat or choropleth; below ~0.7 the basemap reads through.
        cmap: Colormap for ``values``.
        zone_colours: Fill colour of each ``highlight_zones`` entry, in its order.
        zone_edge: Edge colour of the ``highlight_zones`` fills. Setting it to the page colour rather than to
            ``highlight_colour`` is what keeps a marked zone from reading as a selected one.
        zone_edge_width: Width of that edge.
        highlight_colour: Colour of the ``highlight`` outline.
        highlight_width: Width of that outline.
        vmin: Lower bound of the colour scale, shared across panels when set.
        vmax: Upper bound of the colour scale.
        legend: Draw the colour bar and the marker legend.
        basemap: Add the ``BASEMAP`` tiles underneath.
        boundary_width: Width of the zone outlines; thinner reads better over the whole region.
        figsize: Size of the new figure when ``ax`` is None.
        title: Axes title.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if values is None:
        nodes.plot(ax=ax, color=fill, alpha=alpha)
    else:
        nodes.assign(_value=values).plot(
            ax=ax, column="_value", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, legend=legend
        )

    nodes.boundary.plot(ax=ax, color=BOUNDARY_COLOUR, linewidth=boundary_width)

    handles = []

    if highlight_zones:
        for (label, indices), colour in zip(highlight_zones.items(), zone_colours, strict=False):
            nodes.iloc[list(indices)].plot(
                ax=ax, color=colour, edgecolor=zone_edge, linewidth=zone_edge_width, alpha=0.95, zorder=3
            )
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor="none", label=label))

    if edges is not None:
        edges.to_crs(CRS).plot(ax=ax, color=EDGE_COLOUR, linewidth=0.9, alpha=0.85, zorder=4)

    if show_nodes:
        points = _centroids(nodes)
        ax.scatter(points.x, points.y, s=14, color=NODE_COLOUR, edgecolor="white", linewidth=0.4, zorder=5)

    if highlight is not None:
        nodes.iloc[list(highlight)].boundary.plot(
            ax=ax, color=highlight_colour, linewidth=highlight_width, zorder=6
        )

    if markers:
        centroids = _centroids(nodes)

        for (label, indices), colour in zip(markers.items(), MARKER_COLOURS, strict=False):
            points = centroids.iloc[list(indices)]
            scatter = ax.scatter(
                points.x, points.y, s=120, color=colour, edgecolor="white", linewidth=1.2, zorder=7, label=label
            )
            handles.append(scatter)

    if legend and handles:
        ax.legend(handles=handles, **MAP_LEGEND)

    if focus is not None:
        min_x, min_y, max_x, max_y = nodes.iloc[list(focus)].total_bounds
        pad_x, pad_y = (max_x - min_x) * focus_margin, (max_y - min_y) * focus_margin
        min_x, max_x, min_y, max_y = min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y
        anchor = None if aspect_anchor is None else float(_centroids(nodes.iloc[[aspect_anchor]]).y.iloc[0])
        min_y, max_y = _flatten(
            (min_y, max_y), _ground_aspect(nodes.iloc[list(focus)]), max_aspect, anchor
        )
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    if basemap:
        cx.add_basemap(ax, crs=CRS, source=BASEMAP, attribution=False)

    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    return ax


def zone_figsize(nodes: gpd.GeoDataFrame, width: float = 7.0, max_aspect: float | None = None) -> tuple[float, float]:
    """A figure size matching the ground aspect of ``nodes``, so a cropped view is not letterboxed.

    ``zone_map`` draws in lat/lon with an equal-ground-distance aspect, so a figure whose own aspect
    differs from the zones' pads the short side with basemap. The ratio is taken in a projected CRS,
    where it is a true ground ratio.

    Args:
        nodes: The zones that will be drawn.
        width: Figure width in inches.
        max_aspect: Largest height over width the figure may take, matching ``zone_map``'s own.

    Returns:
        ``(width, height)`` in inches.
    """
    aspect = _ground_aspect(nodes)

    return width, width * (aspect if max_aspect is None else min(aspect, max_aspect))


def _ground_aspect(nodes: gpd.GeoDataFrame) -> float:
    """The height over width of the zones' bounding box, measured in a projected CRS."""
    min_x, min_y, max_x, max_y = nodes.to_crs(nodes.estimate_utm_crs()).total_bounds

    return (max_y - min_y) / (max_x - min_x)


def _flatten(
    limits: tuple[float, float], aspect: float, max_aspect: float | None, anchor: float | None
) -> tuple[float, float]:
    """``limits`` narrowed until the view is at most ``max_aspect`` tall, around ``anchor`` or its own centre.

    A window taken around an anchor near one end is slid back inside ``limits``, so the view never runs
    past the zones it was cropped to.

    Args:
        limits: The view's ``(low, high)`` on the value axis.
        aspect: Ground height over width the view has at ``limits``.
        max_aspect: Largest ground aspect to keep; ``limits`` is returned untouched when None or already met.
        anchor: Value the kept window is centred on; the middle of ``limits`` when None.

    Returns:
        The narrowed ``(low, high)``.
    """
    low, high = limits

    if max_aspect is None or aspect <= max_aspect:
        return low, high

    span = (high - low) * max_aspect / aspect
    centre = (low + high) / 2 if anchor is None else min(max(anchor, low + span / 2), high - span / 2)

    return centre - span / 2, centre + span / 2


def map_figure(
    view: gpd.GeoDataFrame,
    width: float = 7.0,
    legend_width: float = 0.0,
    colour_bar_height: float = 0.0,
    max_aspect: float | None = None,
) -> tuple[Figure, Axes, Axes | None]:
    """A figure whose map area is the same size in every panel, with strips kept clear for the keys.

    The panels only line up when they are saved at their own size, with ``bbox_inches=None``: a tight box
    crops back to the keys each panel happens to carry.

    Args:
        view: The zones the map is cropped to, whose ground aspect the map area takes.
        width: Width of the map area in inches.
        legend_width: Inches reserved to the right of the map area.
        colour_bar_height: Inches reserved under the map area.
        max_aspect: Largest height over width the map area may take, matching the map's own ``max_aspect``.

    Returns:
        ``(fig, ax, cax)``: the figure, the map axes, and the colour bar axes when one was reserved.
    """
    map_width, map_height = zone_figsize(view, width=width, max_aspect=max_aspect)
    fig_width, fig_height = map_width + legend_width, map_height + colour_bar_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes((0.0, colour_bar_height / fig_height, map_width / fig_width, map_height / fig_height))

    if colour_bar_height <= 0:
        return fig, ax, None

    bar_height = 0.2 * colour_bar_height / fig_height
    cax = fig.add_axes((0.15 * map_width / fig_width, 2.2 * bar_height, 0.7 * map_width / fig_width, bar_height))

    return fig, ax, cax


def _centroids(nodes: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Zone centroids in ``CRS``, computed in a projected CRS so they land inside their zone."""
    return nodes.geometry.to_crs(nodes.estimate_utm_crs()).centroid.to_crs(CRS)


def add_transit_line(
    ax: Axes,
    nodes: gpd.GeoDataFrame,
    through: Sequence[int],
    extend: float = 1.0,
    colour: str = TRANSIT_COLOUR,
    linewidth: float = 4.0,
    label: str | None = None,
) -> Axes:
    """Draw a straight transit line through the given zones, running in from off-map and back out.

    Each end is extended by ``extend`` times the view's own diagonal, which is what makes the line
    leave the panel on both sides whatever the zones' spacing; the axes limits are restored
    afterwards so the ends fall outside the view rather than widening it.

    Args:
        ax: Axes holding an already-drawn ``zone_map``.
        nodes: The same zones that map was drawn from.
        through: Node indices (into ``nodes``) the line passes through, in order.
        extend: How far past each end to run, in multiples of the view's diagonal.
        colour: Line colour.
        linewidth: Line width.
        label: Legend label; the legend is redrawn when given.

    Returns:
        The axes drawn on.

    Raises:
        ValueError: If the first and last zone of ``through`` are the same, leaving no direction.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    points = _centroids(nodes).iloc[list(through)]
    coords = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])

    direction = coords[-1] - coords[0]
    length = float(np.hypot(*direction))

    if length == 0.0:
        raise ValueError("`through` must start and end at different zones")

    reach = float(np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])) * extend
    step = direction / length * reach

    line = gpd.GeoSeries([LineString([coords[0] - step, *coords, coords[-1] + step])], crs=CRS)

    line.plot(ax=ax, color=colour, linewidth=linewidth, alpha=0.95, zorder=6, capstyle="round")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if label is not None:
        handles = ax.get_legend().legend_handles if ax.get_legend() else []
        labels = [text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else []
        handles.append(plt.Line2D([0], [0], color=colour, linewidth=linewidth))
        labels.append(label)
        ax.legend(handles, labels, **MAP_LEGEND)

    return ax


def score_panels(
    nodes: gpd.GeoDataFrame,
    scores_by_model: dict[str, np.ndarray],
    figsize: tuple[float, float] = (14, 7),
    **kwargs,
) -> Figure:
    """One ``zone_map`` choropleth per model, side by side on a shared colour scale and extent.

    Args:
        nodes: Zones in PyG node order from ``node_gdf``.
        scores_by_model: Model label -> value per node, e.g. ``rank_prominence`` of a score vector.
        figsize: Size of the figure.
        **kwargs: Forwarded to ``zone_map``.

    Returns:
        The figure holding the panels.
    """
    stacked = np.concatenate(list(scores_by_model.values()))
    kwargs.setdefault("vmin", float(stacked.min()))
    kwargs.setdefault("vmax", float(stacked.max()))

    fig, axes = plt.subplots(1, len(scores_by_model), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, (label, values) in zip(axes, scores_by_model.items(), strict=True):
        zone_map(nodes, values=values, ax=ax, title=label, legend=False, **kwargs)

    fig.colorbar(axes[0].collections[0], ax=axes, fraction=0.025, pad=0.02)

    return fig


# =========================================
# Model curves (Altair)
# =========================================

CHART_SIZE = (560, 360)


def chart_style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Light background and readable label sizes, applied to every chart.

    It sets top-level configuration, which Vega-Lite allows only on the outermost chart, so a panel
    destined for a concatenation is built with ``style=False`` and styled once it is concatenated.
    """
    return (
        chart
        .configure(background="white")
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=12, titleFontSize=13, grid=True, gridColor="#eceff1")
        .configure_legend(labelFontSize=12, titleFontSize=13)
    )


def _stylable(style: bool):
    """``chart_style`` when the chart is the outermost one, and the identity when it is a panel."""
    return chart_style if style else (lambda chart: chart)


def recall_chart(
    curve: pl.DataFrame,
    specs: Sequence[ModelStyle] = DEFAULT_MODELS,
    size: tuple[int, int] = CHART_SIZE,
    tick_ks: Sequence[int] | None = None,
    points: bool = True,
    title: str | list[str] | None = None,
    style: bool = True,
    log_k: bool = True,
) -> alt.LayerChart:
    """Recall@k at the reported cutoffs, one line per model with a +- sd band over the training seeds.

    A tick is forced at each labelled cutoff so they read as the actual values rather than as
    whatever round numbers the scale would pick.

    Args:
        curve: ``analysis.recall_curve`` output, with ``name``, ``k``, ``mean`` and ``sd``.
        specs: Models to draw, in legend order.
        size: Chart ``(width, height)``.
        tick_ks: Cutoffs to label on the axis; every cutoff in ``curve`` when None. A curve drawn over
            a dense set of cutoffs needs a sparser set of ticks than it has points.
        points: Mark each cutoff with a point, which a dense curve does not want.
        title: Chart title, a list for several lines.
        style: Apply ``chart_style``; False for a panel that will be concatenated and styled after.
        log_k: Space ``k`` logarithmically, which spreads the small cutoffs a reported set is dense in;
            False spaces it linearly.

    Returns:
        The layered chart.
    """
    data = with_labels(curve, specs).with_columns(
        lo=pl.col("mean") - pl.col("sd").fill_null(0.0), hi=pl.col("mean") + pl.col("sd").fill_null(0.0)
    )
    ks = sorted(data["k"].unique().to_list())
    axis = alt.X(
        "k:Q",
        title="k",
        scale=alt.Scale(type="log" if log_k else "linear", nice=False, domain=[min(ks), max(ks)]),
        axis=alt.Axis(
            values=list(tick_ks) if tick_ks is not None else ks,
            format="d",
            labelAngle=0,
            labelOverlap=False,
            grid=True,
        ),
    )
    colour = model_colour(specs)

    band = (
        alt
        .Chart(data)
        .mark_area(opacity=0.18, stroke=None)
        .encode(x=axis, y=alt.Y("lo:Q", title="recall@k"), y2="hi:Q", color=colour, detail="model:N")
    )
    line = (
        alt
        .Chart(data)
        .mark_line(point=points, strokeWidth=2)
        .encode(x=axis, y=alt.Y("mean:Q", title="recall@k"), color=colour, strokeDash=model_dash(specs))
    )
    properties = {"width": size[0], "height": size[1]}

    if title is not None:
        properties["title"] = title

    return _stylable(style)(alt.layer(band, line).properties(**properties))


def roc_chart(
    curve: pl.DataFrame,
    n_nodes: int,
    specs: Sequence[ModelStyle] = ROC_MODELS,
    size: tuple[int, int] = CHART_SIZE,
    style: bool = True,
) -> alt.LayerChart:
    """Recall@k over the full range of k, against the share of the region selected.

    Args:
        curve: ``analysis.recall_curve`` output over a dense set of cutoffs.
        n_nodes: Nodes in the region, which turns each ``k`` into a share.
        specs: Models to draw, in legend order.
        size: Chart ``(width, height)``.

    Returns:
        The layered chart, with the random-ranking diagonal underneath.
    """
    data = with_labels(curve, specs).with_columns(share=pl.col("k") / n_nodes)

    diagonal = (
        alt
        .Chart(pl.DataFrame({"share": [0.0, 1.0], "mean": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="#9aa0a6", strokeWidth=1)
        .encode(x="share:Q", y="mean:Q")
    )
    line = (
        alt
        .Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("share:Q", title="share of the region selected (k / n)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("mean:Q", title="recall@k", scale=alt.Scale(domain=[0, 1])),
            color=model_colour(specs),
        )
    )

    return _stylable(style)(alt.layer(diagonal, line).properties(width=size[0], height=size[1]))


BAND_COUNT_COLOUR = "#8c959f"
BAND_COUNT_OFFSET = 26
"""How far under the plot area the band's positives are written. The axis title is pushed down by as much,
since a text mark is not part of the axis and does not move it."""


def band_axis_labels(band_table: pl.DataFrame) -> pl.DataFrame:
    """Add each band's axis label, its range in km, and the count of positives written under it.

    Args:
        band_table: ``analysis.distance_band_recall_table`` output.

    Returns:
        ``band_table`` with ``band_label`` and ``band_positives`` columns.
    """
    return band_table.with_columns(
        band_label=pl.col("band") + pl.lit(" km"),
        band_positives=(
            pl.col("n_pos").round(0).cast(pl.Int64).cast(pl.String)
            + pl.lit(" visits (")
            + (100 * pl.col("share_of_pos")).round(0).cast(pl.Int64).cast(pl.String)
            + pl.lit("%)")
        ),
    )


def band_recall_chart(
    band_table: pl.DataFrame,
    budget: int,
    specs: Sequence[ModelStyle] = DEFAULT_MODELS,
    size: tuple[int, int] = CHART_SIZE,
    title: str | list[str] | None = None,
    style: bool = True,
) -> alt.LayerChart:
    """Recall of the visited locations in each distance band from home, one line per model.

    Each band's positives are written in grey under its distance, which is the label a reader of the axis
    is after.

    Args:
        band_table: ``analysis.distance_band_recall_table`` output.
        budget: The ``K`` the per-user ``avg_recall`` was derived at, named in the value axis title.
        specs: Models to draw, in legend order.
        size: Chart ``(width, height)``.
        title: Chart title, a list for several lines.
        style: Apply ``chart_style``; False for a panel that will be concatenated and styled after.

    Returns:
        The layered chart.
    """
    data = with_labels(band_axis_labels(band_table), specs)
    axis = alt.X(
        "band_label:N",
        title="straight-line distance from home",
        sort=data["band_label"].unique(maintain_order=True).to_list(),
        axis=alt.Axis(labelAngle=0, titlePadding=BAND_COUNT_OFFSET),
    )
    line = (
        alt
        .Chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=axis,
            y=alt.Y(
                "mean:Q",
                title=f"avg_recall@{budget} of the visits in the band",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=model_colour(specs),
            strokeDash=model_dash(specs),
        )
    )
    positives = (
        alt
        .Chart(data.select("band_label", "band_positives").unique(maintain_order=True))
        .mark_text(dy=BAND_COUNT_OFFSET, color=BAND_COUNT_COLOUR, fontSize=9, baseline="top")
        .encode(x=axis, y=alt.value(size[1]), text="band_positives:N")
    )
    properties = {"width": size[0], "height": size[1]}

    if title is not None:
        properties["title"] = title

    return _stylable(style)(alt.layer(line, positives).properties(**properties))


# =========================================
# Tables
# =========================================


def recall_auc_table(curve: pl.DataFrame, n_nodes: int, specs: Sequence[ModelStyle] = ROC_MODELS) -> pl.DataFrame:
    """Area under each model's recall@k curve, read against the share of the region selected.

    The curve is anchored at ``(0, 0)`` and integrated by the trapezoid rule, so 1.0 is a ranking
    that recovers every positive at the very top and 0.5 is the random diagonal.

    Args:
        curve: ``analysis.recall_curve`` output over a dense set of cutoffs.
        n_nodes: Nodes in the region, which turns each ``k`` into a share.
        specs: Models to report, in scheme order.

    Returns:
        Frame of ``model, auc``, ordered by ``auc`` descending.
    """
    data = with_labels(curve, specs).with_columns(share=pl.col("k") / n_nodes).sort("share")

    rows = [
        {
            "model": group["model"][0],
            "auc": float(
                np.trapezoid(
                    np.concatenate([[0.0], group["mean"].to_numpy()]),
                    np.concatenate([[0.0], group["share"].to_numpy()]),
                )
            ),
        }
        for _, group in data.group_by("model", maintain_order=True)
    ]

    return pl.DataFrame(rows).sort("auc", descending=True)


def avg_recall_table(
    per_user_results: pl.DataFrame,
    budget: int,
    specs: Sequence[ModelStyle] = ROC_MODELS,
) -> pl.DataFrame:
    """Each model's ``avg_recall@budget``, mean +- sd over the training seeds.

    Args:
        per_user_results: Per-user results frame carrying the ``avg_recall`` column.
        budget: The ``K`` the per-user ``avg_recall`` was derived at, used only for the column name.
        specs: Models to report, in scheme order.

    Returns:
        Frame of ``model, n_users, n_seeds, mean, sd``, ordered by mean descending.
    """
    summary = per_user_metric_summary(per_user_results, AVG_RECALL_COLUMN)

    return (
        with_labels(summary, specs)
        .select("model", "n_users", "n_seeds", "mean", "sd")
        .rename({"mean": f"avg_recall@{budget}"})
        .sort(f"avg_recall@{budget}", descending=True)
    )
