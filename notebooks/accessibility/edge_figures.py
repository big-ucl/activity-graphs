"""Figures of the step 6 reads: the paired band reads as heatmaps, and the change from the rail edges on a map.

The marimo notebook ``edge_figures_nb.py`` shows them. The heatmaps show, for every model and band, the paired
difference against a reference model: in the within-band AUC, in the recall by distance band and in the recall by hop
band. The maps show, for every zone, the mean change in recall between two models, either of the test positives in the
zone or of the test users living in it.

The per-user within-band AUC of a run takes minutes on TTS, so it is cached under ``CACHE_DIR``, one file per run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import altair as alt
import branca.colormap
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely
from omegaconf import DictConfig, OmegaConf

from activitygraphs.analysis import (
    distance_band_recall,
    paired_comparison_by_band,
    paired_distance_band_recall,
    paired_within_band_auc,
    read_home_distances,
    read_report_run,
    read_score_vectors,
    within_band_auc,
)
from activitygraphs.base import CRS, PER_USER_STAGE
from activitygraphs.graph import GraphTables
from activitygraphs.graph.build import GRAPH_DIR
from activitygraphs.runs import CONFIG_FILE, run_dir
from activitygraphs.schema import EdgeKind

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "reports" / "logs" / "edge-figures"

RUNS = {"tts2016": 21, "cmap": 28, "tpg": 11}
METRICS = ("within-band AUC", "recall by distance band", "recall by hop band")
N_BOOTSTRAP = 1000
DEFAULT_MIN_ZONE_COUNT = 20
DEFAULT_HEX_KM = 5.0
METRES_PER_KM = 1000.0
Z_95 = 1.96
SIMPLIFY_TOLERANCE_M = 25.0

GAIN_COLOR = "#2a78d6"
NEUTRAL_COLOR = "#f0efec"
LOSS_COLOR = "#e34948"
MASKED_COLOR = "#d9d8d4"
STATION_COLOR = "#2b2b29"

BASEMAP_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_ATTRIBUTION = "Tiles (C) Esri -- Esri, DeLorme, NAVTEQ"
"""Esri's light grey canvas, served with no API key. The CARTO basemaps watermark the tiles requested without one."""


@dataclass(frozen=True)
class RunReads:
    """The reads of one recorded run that the figures draw from.

    Attributes:
        data: Dataset key, e.g. ``"tts2016"``.
        run: Run number.
        names: Name of every model of the run, trained models first.
        per_user: Per-user results with the ranking metrics, from ``read_report_run``.
        band_recall: Per-user recall by distance band, from ``distance_band_recall``.
        band_auc: Per-user within-band AUC, from ``within_band_auc``.
        zones: Zones of the graph in node order, with their polygon and their rail degree, the number of zones a rail
            edge joins each zone to.
    """

    data: str
    run: int
    names: list[str]
    per_user: pl.DataFrame
    band_recall: pl.DataFrame
    band_auc: pl.DataFrame
    zones: gpd.GeoDataFrame


def load_run_reads(data: str, run: int) -> RunReads:
    """Return the reads of run ``run`` of the dataset ``data``, with the within-band AUC read from the cache if there."""
    run_path = _run_path(data, run)
    cfg = cast(DictConfig, OmegaConf.load(run_path / CONFIG_FILE))
    bands = list(cfg.data.distance_bands)

    _, per_user = read_report_run(run_path, cfg.train.max_recall_k)
    home_distances = read_home_distances(run_path)
    band_recall = distance_band_recall(per_user, home_distances, bands)

    cache_path = CACHE_DIR / f"{run_path.name}-band-auc.parquet"
    if cache_path.exists():
        band_auc = pl.read_parquet(cache_path)
    else:
        band_auc = within_band_auc(per_user, read_score_vectors(run_path), home_distances, bands)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        band_auc.write_parquet(cache_path)

    test_rows = per_user.filter(pl.col("stage") == PER_USER_STAGE)
    trained_names = sorted(test_rows.filter(pl.col("seed").is_not_null())["name"].unique())
    baseline_names = sorted(set(test_rows["name"].unique()) - set(trained_names))
    graph = GraphTables.read(PROJECT_ROOT / cfg.data.paths.processed / GRAPH_DIR)

    return RunReads(
        data=data,
        run=run,
        names=[*trained_names, *baseline_names],
        per_user=per_user,
        band_recall=band_recall,
        band_auc=band_auc,
        zones=_zone_shapes(graph),
    )


# =========================================
# Heatmaps of the band reads
# =========================================


def band_differences(reads: RunReads, reference: str, models: list[str], metric: str) -> pl.DataFrame:
    """Return the paired difference of every model against ``reference`` in each band of ``metric``.

    Args:
        reads: The reads of the run.
        reference: Name of the reference model.
        models: Names of the models to compare against ``reference``.
        metric: One of ``METRICS``.

    Returns:
        Frame of ``band, band_order, name, mean_diff, ci_lo, ci_hi, is_resolved``. ``band_order`` orders the bands from
        home outwards. A difference is resolved when its 95% bootstrap interval excludes zero.
    """
    kept_names = [*models, reference]
    if metric == METRICS[0]:
        band_auc = reads.band_auc.filter(pl.col("name").is_in(kept_names))
        differences = paired_within_band_auc(band_auc, reference, N_BOOTSTRAP)
        band_order = _distance_band_order(reads.band_auc)
    elif metric == METRICS[1]:
        band_recall = reads.band_recall.filter(pl.col("name").is_in(kept_names))
        differences = paired_distance_band_recall(band_recall, reference, N_BOOTSTRAP)
        band_order = _distance_band_order(reads.band_recall)
    elif metric == METRICS[2]:
        per_user = reads.per_user.filter(pl.col("name").is_in(kept_names))
        differences = paired_comparison_by_band(per_user, reference, "avg_recall", N_BOOTSTRAP)
        band_order = _hop_band_order(differences)
    else:
        raise KeyError(f"No metric `{metric}`; the metrics are {METRICS}.")

    is_resolved = (pl.col("ci_lo") > 0) | (pl.col("ci_hi") < 0)

    return (
        differences
        .join(band_order, on="band")
        .with_columns(is_resolved=is_resolved)
        .select("band", "band_order", "name", "mean_diff", "ci_lo", "ci_hi", "is_resolved")
    )


def band_heatmap(differences: pl.DataFrame, reference: str, metric: str) -> alt.LayerChart | alt.FacetChart:
    """Return a heatmap of ``differences``: one row per model, one column per band, the colour the paired difference.

    A dot marks every difference whose 95% interval excludes zero.

    Args:
        differences: Paired differences from ``band_differences``.
        reference: Name of the reference model, for the title.
        metric: One of ``METRICS``, for the title.
    """
    largest = float(np.abs(differences["mean_diff"].to_numpy()).max(initial=1e-6))
    band_sort = alt.EncodingSortField(field="band_order", order="ascending")
    model_order = differences.group_by("name").agg(pl.col("mean_diff").mean()).sort("mean_diff", descending=True)
    x = alt.X("band:N", sort=band_sort, title="band from home", axis=alt.Axis(labelAngle=0))
    y = alt.Y("name:N", sort=model_order["name"].to_list(), title=None)
    tooltip = [
        alt.Tooltip("name:N", title="model"),
        alt.Tooltip("band:N"),
        alt.Tooltip("mean_diff:Q", title=f"difference vs {reference}", format="+.4f"),
        alt.Tooltip("ci_lo:Q", title="95% CI low", format="+.4f"),
        alt.Tooltip("ci_hi:Q", title="95% CI high", format="+.4f"),
    ]
    color_scale = alt.Scale(
        domain=[-largest, 0, largest], range=[LOSS_COLOR, NEUTRAL_COLOR, GAIN_COLOR], interpolate="lab"
    )

    cells = (
        alt
        .Chart(differences)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(x=x, y=y, color=alt.Color("mean_diff:Q", scale=color_scale, title="difference"), tooltip=tooltip)
    )
    resolved = (
        alt
        .Chart(differences.filter("is_resolved"))
        .mark_point(shape="circle", filled=True, size=40, color=STATION_COLOR)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    title = f"{metric}: paired difference vs {reference} (dot: 95% CI excludes 0)"

    return (cells + resolved).properties(title=title, width="container", height=alt.Step(26))


# =========================================
# Maps of the change between two models
# =========================================


def zone_changes(reads: RunReads, model: str, reference: str, by: str, min_count: int) -> gpd.GeoDataFrame:
    """Return the mean paired change in recall from ``reference`` to ``model`` in every zone.

    Args:
        reads: The reads of the run.
        model: Name of the model.
        reference: Name of the reference model.
        by: ``"destination"`` to average the per-positive ``avg_recall`` of the test positives in each zone, or
            ``"home"`` to average the per-user ``avg_recall`` of the test users living in each zone.
        min_count: Number of positives or users a zone needs for its mean change to be shown.

    Returns:
        One row per zone, following ``_cell_frame``, with ``cell`` the zone's ``loc_id``. Each model is averaged over
        its seeds before the change is taken.
    """
    test_rows = reads.per_user.filter(pl.col("stage") == PER_USER_STAGE, pl.col("name").is_in([model, reference]))
    if by == "destination":
        values = (
            test_rows
            .select("name", "user_idx", "pos_node", "pos_avg_recall")
            .explode("pos_node", "pos_avg_recall")
            .rename({"pos_node": "node_idx", "pos_avg_recall": "value"})
        )
    elif by == "home":
        values = test_rows.select("name", "user_idx", node_idx="home_node", value="avg_recall")
    else:
        raise KeyError(f"`by` is `destination` or `home`, got `{by}`.")

    seed_means = values.group_by("name", "user_idx", "node_idx").agg(pl.col("value").mean())
    paired = (
        seed_means
        .filter(pl.col("name") == model)
        .join(seed_means.filter(pl.col("name") == reference), on=["user_idx", "node_idx"], suffix="_reference")
        .with_columns(change=pl.col("value") - pl.col("value_reference"))
    )
    zone_sums = paired.group_by("node_idx").agg(
        count=pl.len(), sum_change=pl.col("change").sum(), sum_sq_change=(pl.col("change") ** 2).sum()
    )
    zone_frame = (
        pl
        .DataFrame({"node_idx": np.arange(len(reads.zones))})
        .join(zone_sums, on="node_idx", how="left")
        .fill_null(0)
        .sort("node_idx")
    )

    zones = reads.zones.copy()
    zones["count"] = zone_frame["count"].to_numpy()
    zones["sum_change"] = zone_frame["sum_change"].to_numpy()
    zones["sum_sq_change"] = zone_frame["sum_sq_change"].to_numpy()

    return _cell_frame(zones.rename(columns={"loc_id": "cell"}), min_count)


def hex_changes(zones: gpd.GeoDataFrame, hex_km: float, min_count: int) -> gpd.GeoDataFrame:
    """Return the mean paired change of the zones of ``zones`` pooled into a grid of hexagons ``hex_km`` across.

    A zone falls in the hexagon that holds its centre. The change of a hexagon is the mean over every positive or user
    of its zones, not a mean of the zone means.

    Args:
        zones: Zone changes from ``zone_changes``.
        hex_km: Width of a hexagon from one flat side to the opposite one, in km.
        min_count: Number of positives or users a hexagon needs for its mean change to be shown.

    Returns:
        One row per hexagon holding at least one zone centre, following ``_cell_frame``.
    """
    utm_crs = zones.estimate_utm_crs()
    projected_zones = zones.to_crs(utm_crs)
    centres = gpd.GeoDataFrame(
        projected_zones[["count", "sum_change", "sum_sq_change", "rail_degree"]],
        geometry=projected_zones.centroid,
        crs=utm_crs,
    )
    hexagons = _hex_grid(centres.total_bounds, hex_km * METRES_PER_KM, utm_crs)
    zone_cells = gpd.sjoin(centres, hexagons, predicate="within")

    cell_sums = zone_cells.groupby("cell")[["count", "sum_change", "sum_sq_change"]].sum()
    cell_degrees = zone_cells.groupby("cell")["rail_degree"].max()
    cells = hexagons.set_index("cell").join(cell_sums, how="inner").join(cell_degrees).reset_index()

    return _cell_frame(cells.to_crs(CRS), min_count)


def zone_map(cells: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, model: str, reference: str, by: str) -> folium.Map:
    """Return a zoomable map of ``cells``: every zone or hexagon coloured by its mean change, with the stations marked.

    Cells not shown are grey. A dark outline marks a cell whose 95% interval excludes zero. A ring marks the centre of
    every station zone, sized by its rail degree. The colour scale stops at the 95th percentile of the absolute changes
    shown. Larger changes take the end colours.

    Args:
        cells: Cell changes from ``zone_changes`` or ``hex_changes``.
        stations: The zones of ``RunReads.zones`` that have a rail edge.
        model: Name of the model, for the legend.
        reference: Name of the reference model, for the legend.
        by: ``"destination"`` or ``"home"``, for the legend and the tooltips.
    """
    shown = cells[cells["is_shown"]].copy()
    hidden = cells[~cells["is_shown"]]
    largest = float(np.nanquantile(np.abs(shown["mean_change"]), 0.95)) if len(shown) else 1e-6
    shown["mean_change"] = shown["mean_change"].round(4)
    shown["ci_half_width"] = shown["ci_half_width"].round(4)
    count_label = "positives" if by == "destination" else "users"
    caption = f"{model} minus {reference}: mean change in recall by {by} (grey: too few {count_label})"
    colormap = branca.colormap.LinearColormap(
        [LOSS_COLOR, NEUTRAL_COLOR, GAIN_COLOR], vmin=-largest, vmax=largest, caption=caption
    )

    zone_map = folium.Map(tiles=BASEMAP_TILES, attr=BASEMAP_ATTRIBUTION, height=650)
    min_lon, min_lat, max_lon, max_lat = cells.total_bounds
    zone_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    cell_style = {"color": "white", "weight": 0.4, "fillOpacity": 0.75}

    if len(hidden):
        hidden_cells = hidden[["cell", "count", "rail_degree", "geometry"]]
        hidden_cells.explore(
            m=zone_map,
            color=MASKED_COLOR,
            tooltip=True,
            tooltip_kwds={"aliases": ["cell", count_label, "highest rail degree"]},
            style_kwds={**cell_style, "fillOpacity": 0.5},
        )
    if len(shown):
        shown_cells = shown[["cell", "mean_change", "ci_half_width", "count", "rail_degree", "geometry"]]
        shown_cells.explore(
            m=zone_map,
            column="mean_change",
            cmap=colormap,
            legend=False,
            tooltip=True,
            tooltip_kwds={"aliases": ["cell", "mean change", "95% CI +-", count_label, "highest rail degree"]},
            style_kwds=cell_style,
        )
    resolved = shown[shown["is_resolved"]]
    if len(resolved):
        resolved_outlines = resolved[["geometry"]]
        resolved_outlines.explore(
            m=zone_map,
            tooltip=False,
            style_kwds={"color": STATION_COLOR, "weight": 1.8, "fill": False},
        )

    for loc_id, degree, lon, lat in stations[["loc_id", "rail_degree", "centroid_lon", "centroid_lat"]].itertuples(
        index=False
    ):
        ring_radius = 2 + float(np.sqrt(degree))
        folium.CircleMarker(
            location=[lat, lon],
            radius=ring_radius,
            color=STATION_COLOR,
            weight=1.5,
            fill=False,
            tooltip=f"zone {loc_id}: rail degree {degree}",
        ).add_to(zone_map)

    colormap.add_to(zone_map)

    return zone_map


def degree_scatter(cells: gpd.GeoDataFrame, model: str, reference: str) -> alt.Chart:
    """Return a scatter of the mean change of every shown cell against its highest rail degree, sized by its count.

    Cells whose 95% interval excludes zero are dark.
    """
    shown = pl.from_pandas(pd.DataFrame(cells.loc[cells["is_shown"]].drop(columns="geometry")))
    tooltip = [
        alt.Tooltip("cell:N"),
        alt.Tooltip("rail_degree:Q", title="highest rail degree"),
        alt.Tooltip("mean_change:Q", title="mean change", format="+.4f"),
        alt.Tooltip("ci_half_width:Q", title="95% CI +-", format=".4f"),
        alt.Tooltip("count:Q"),
    ]
    resolved_color = alt.Color(
        "is_resolved:N",
        scale=alt.Scale(domain=[False, True], range=[MASKED_COLOR, GAIN_COLOR]),
        title="95% CI excludes 0",
    )

    return (
        alt
        .Chart(shown)
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X(
                "rail_degree:Q",
                scale=alt.Scale(type="symlog"),
                axis=alt.Axis(values=[0, 1, 2, 5, 10, 20, 50, 100, 200], format="d", grid=False),
                title="highest rail degree in the cell (symlog)",
            ),
            y=alt.Y("mean_change:Q", title=f"mean change, {model} minus {reference}"),
            size=alt.Size("count:Q", scale=alt.Scale(range=[10, 300]), title="count"),
            color=resolved_color,
            tooltip=tooltip,
        )
        .properties(title="mean change by rail degree", width="container", height=320)
    )


def _run_path(data: str, run: int) -> Path:
    """Return the directory of run ``run`` of the dataset ``data``, under the default runs path."""
    cfg = OmegaConf.load(PROJECT_ROOT / "activitygraphs" / "conf" / "data" / f"{data}.yaml")
    return run_dir(PROJECT_ROOT / "reports" / "runs", cfg.name, run)


def _zone_shapes(graph: GraphTables) -> gpd.GeoDataFrame:
    """Return the zones of ``graph`` in node order, with a simplified polygon, the centre and the rail degree.

    Every polygon part is wound clockwise and every hole counterclockwise, the orientation read by Vega's map
    projection. A part wound the other way covers the whole globe but itself.
    """
    nodes = graph.nodes
    polygons = nodes.set_geometry("original_geometry").geometry
    utm_crs = polygons.estimate_utm_crs()
    simplified = polygons.to_crs(utm_crs).simplify(SIMPLIFY_TOLERANCE_M).to_crs(CRS)
    clockwise = shapely.orient_polygons(simplified.to_numpy(), exterior_cw=True)
    centroids = polygons.to_crs(utm_crs).centroid.to_crs(CRS)

    rail_edges = graph.edges[EdgeKind.RAIL].index
    rail_ends = [*rail_edges.get_level_values("source").astype(str), *rail_edges.get_level_values("target").astype(str)]
    rail_degree = pl.DataFrame({"loc_id": rail_ends}).group_by("loc_id").len("rail_degree")
    loc_ids = pl.DataFrame({"loc_id": nodes.index.astype(str).to_list()})
    degree_by_zone = loc_ids.join(rail_degree, on="loc_id", how="left")

    return gpd.GeoDataFrame(
        {
            "loc_id": loc_ids["loc_id"].to_numpy(),
            "rail_degree": degree_by_zone["rail_degree"].fill_null(0).to_numpy(),
            "centroid_lon": centroids.x.to_numpy(),
            "centroid_lat": centroids.y.to_numpy(),
        },
        geometry=clockwise,
        crs=CRS,
    )


def _distance_band_order(band_frame: pl.DataFrame) -> pl.DataFrame:
    """Return the order of every distance band of ``band_frame``, from home outwards, as ``band, band_order``."""
    return band_frame.select("band", band_order="band_lo").unique()


def _hop_band_order(differences: pl.DataFrame) -> pl.DataFrame:
    """Return the order of every hop band of ``differences``, from home outwards, as ``band, band_order``."""
    lower_hops = pl.col("band").str.extract(r"^(\d+)").cast(pl.Float64)
    return differences.select("band").unique().with_columns(band_order=lower_hops)


def _cell_frame(cells: gpd.GeoDataFrame, min_count: int) -> gpd.GeoDataFrame:
    """Return the mean change and its interval in every cell of ``cells``, from the sums of the changes in the cell.

    Args:
        cells: One row per cell with ``cell``, ``count``, ``sum_change``, ``sum_sq_change``, ``rail_degree`` and the
            cell's geometry.
        min_count: Number of positives or users a cell needs for its mean change to be shown.

    Returns:
        Frame of ``cell, mean_change, ci_half_width, count, sum_change, sum_sq_change, rail_degree, is_shown,
        is_resolved`` and the geometry. The sums are kept so that cells can be pooled exactly. ``ci_half_width`` is half the width of the normal 95% interval of the mean. It treats the changes as
        independent, though the positives of one user are not. A cell is resolved when it is shown and its interval
        excludes zero.
    """
    count = cells["count"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_change = cells["sum_change"].to_numpy() / count
        variance = cells["sum_sq_change"].to_numpy() / count - mean_change**2
        ci_half_width = Z_95 * np.sqrt(np.clip(variance, 0, None) / count)

    is_shown = count >= min_count
    is_resolved = is_shown & (np.abs(mean_change) > ci_half_width)

    return gpd.GeoDataFrame(
        {
            "cell": cells["cell"].astype(str).to_numpy(),
            "mean_change": mean_change,
            "ci_half_width": ci_half_width,
            "count": cells["count"].to_numpy(),
            "sum_change": cells["sum_change"].to_numpy(),
            "sum_sq_change": cells["sum_sq_change"].to_numpy(),
            "rail_degree": cells["rail_degree"].to_numpy(),
            "is_shown": is_shown,
            "is_resolved": is_resolved,
        },
        geometry=cells.geometry.to_numpy(),
        crs=cells.crs,
    )


def _hex_grid(bounds: np.ndarray, width_m: float, crs: Any) -> gpd.GeoDataFrame:
    """Return a grid of pointy-topped hexagons ``width_m`` across that covers ``bounds``, in ``crs``.

    Returns:
        Frame of ``cell``, a label per hexagon, and the hexagon's polygon.
    """
    min_x, min_y, max_x, max_y = bounds
    radius = width_m / np.sqrt(3)
    row_step = 1.5 * radius
    corner_angles = np.deg2rad(np.arange(30, 390, 60))
    corners = np.stack([np.cos(corner_angles), np.sin(corner_angles)], axis=1) * radius

    labels, polygons = [], []
    num_rows = int(np.ceil((max_y - min_y) / row_step)) + 2
    num_columns = int(np.ceil((max_x - min_x) / width_m)) + 2
    for row in range(num_rows):
        centre_y = min_y - radius + row * row_step
        row_offset = width_m / 2 if row % 2 else 0.0
        for column in range(num_columns):
            centre_x = min_x - width_m + column * width_m + row_offset
            labels.append(f"hex {row}-{column}")
            polygons.append(shapely.Polygon(corners + [centre_x, centre_y]))

    return gpd.GeoDataFrame({"cell": labels}, geometry=polygons, crs=crs)
