"""Public transport travel times between every pair of zones, routed by R5 over the OSM extract and the GTFS feeds."""

import datetime as dt
import logging
import sys
from collections.abc import Hashable
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
import shapely

from activitygraphs.base import CRS
from activitygraphs.graph.rail import EDGE_INDEX_NAMES
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema

log = logging.getLogger(__name__)

PT_DEPARTURE = dt.time(7, 0)
"""Start of the window of departures over which the median travel time is taken."""

PT_DEPARTURE_WINDOW = dt.timedelta(hours=2)
"""Length of the window of departures, one departure every minute."""

MAX_TRAVEL_TIME = dt.timedelta(minutes=180)
"""Maximum travel time in routing. Slower & impossible journeys get assigned this time and are flagged unreachable."""

WALK_SPEED_KMH = 4.8
"""Walking speed of the walks to/from/between stops, and of journeys only made on foot."""

NON_WALKABLE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "construction",
    "proposed",
    "raceway",
    "bus_guideway",
    "busway",
}
"""Excluded values of OSM ways attribute ``highway``, for the snapping of a zone point."""

NO_ACCESS_VALUES = {"no", "private"}
"""Excluded values of OSM ways attributes ``foot`` and ``access``, that cannot be walked over."""

FOOT_ACCESS_VALUES = {"yes", "designated", "permissive"}
"""Allowed values of OSM ways attribute ``foot``, that allow walking regardless of the ``access`` value."""

WALKABLE_SEARCH_RADIUS_DEG = 0.02
"""Half-width in degrees of the box around a zone point searched for walkable ways, about 2 km."""

TRAVEL_MODES = ("pt", "car", "walk", "bike")
"""Modes of the travel times between the zones, in tensor column order."""

TRAVEL_TIME_SCHEMA = TensorSchema(
    "travel_times",
    (
        *(ColumnSpec(f"{mode}_time_min", ColumnKind.COUNT) for mode in TRAVEL_MODES),
        *(ColumnSpec(f"{mode}_reachable", ColumnKind.BINARY) for mode in TRAVEL_MODES),
    ),
)
"""Schema of the travel-time tensor. The columns are the time in minutes of each mode, then the reachability flag of each
mode."""


def build_pt_travel_times(
    nodes_gdf: gpd.GeoDataFrame, osm_path: Path, feed_paths: list[Path], service_date: dt.date
) -> pd.DataFrame:
    """Return the public transport travel time between every pair of zones on the morning of the service date.

    For each pair of zones, the travel time is computed as the median travel time over ever departure time in the
    window (every minute), by PT and walking. Travel time is computed from the origin zone point to the destination
    zone point. Zone points are snapped to a walkable street by R5. Zones that cannot then reach another zone is
    snapped instead to the nearest walkable OSM way, and routed again.

    Args:
        nodes_gdf: Zones indexed by ``loc_id``, with the zone point in ``geometry``.
        osm_path: The OSM extract of the street network.
        feed_paths: The zipped GTFS feeds.
        service_date: The date of the departures.

    Returns:
        One row per ordered pair of zones, indexed by the (``source``, ``target``) ``loc_id`` of the origin and the
        destination, with the travel time in minutes in ``pt_time_min`` and whether the destination is reached within
        ``MAX_TRAVEL_TIME`` in ``pt_reachable``.
    """
    # r5py starts a JVM on import, so import here only when called.
    import r5py

    _restore_jpype_import_finder()

    network = r5py.TransportNetwork(osm_path, feed_paths)
    departure = dt.datetime.combine(service_date, PT_DEPARTURE)

    zone_points = nodes_gdf[["geometry"]].to_crs(CRS)
    r5_snapped_geometry = network.snap_to_network(zone_points.geometry, street_mode=r5py.TransportMode.WALK)
    zone_points["geometry"] = r5_snapped_geometry.where(~r5_snapped_geometry.is_empty, zone_points.geometry)
    zone_points["id"] = zone_points.index

    travel_times = _route(r5py, network, zone_points, zone_points, departure)
    unlinked_ids = _unlinked_zone_ids(travel_times)

    if unlinked_ids:
        unlinked_centroids = nodes_gdf.loc[unlinked_ids].geometry.to_crs(CRS)
        moved_geometry, moved_distance_m = _nearest_walkable_points(unlinked_centroids, osm_path)
        moved_distance_list = np.round(moved_distance_m).astype(int).tolist()
        moved_distances = dict(zip(unlinked_ids, moved_distance_list, strict=True))
        log.info("Zone points moved to the nearest walkable OSM way, in metres: %s", moved_distances)

        zone_points.loc[unlinked_ids, "geometry"] = moved_geometry.to_numpy()
        moved_points = zone_points.loc[unlinked_ids]
        from_moved = _route(r5py, network, moved_points, zone_points, departure)
        to_moved = _route(r5py, network, zone_points, moved_points, departure)
        travel_times = _replace_pairs(travel_times, [from_moved, to_moved])

    still_unlinked_ids = _unlinked_zone_ids(travel_times)
    if still_unlinked_ids:
        log.warning("Some zones cannot reach any other zone by public transport or on foot: %s", still_unlinked_ids)

    return travel_times


def walkable_candidate_points(
    points: gpd.GeoSeries, osm_path: Path, max_candidates: int
) -> dict[Hashable, tuple[gpd.GeoSeries, np.ndarray]]:
    """For each point of ``points``, return the closest point on each walkable OSM way near it, the closest first.

    The ways searched are those within ``WALKABLE_SEARCH_RADIUS_DEG`` of the point.

    Args:
        points: Points in ``CRS``.
        osm_path: The OSM extract read for the ways.
        max_candidates: Largest number of ways, and so of closest points, returned for one point.

    Returns:
        Dict keyed by the index label of each point of ``points``. Each value is a tuple
        ``(candidate_points, distance_m)``: the closest point on each of the ``max_candidates`` nearest ways, in
        ``CRS`` and the closest first, and the distance in metres from the point of ``points`` to each of them.
    """
    point_x = points.x.to_numpy()
    point_y = points.y.to_numpy()
    radius = WALKABLE_SEARCH_RADIUS_DEG
    search_boxes = shapely.box(point_x - radius, point_y - radius, point_x + radius, point_y + radius)
    walkable_ways = gpd.GeoSeries(_read_walkable_ways(osm_path, shapely.union_all(search_boxes)), crs=CRS)

    utm_crs = points.estimate_utm_crs()
    utm_points = points.to_crs(utm_crs)
    utm_ways = walkable_ways.to_crs(utm_crs).to_numpy()

    candidates = {}
    for label, utm_point, search_box in zip(points.index, utm_points, search_boxes, strict=True):
        ways_in_box = utm_ways[walkable_ways.intersects(search_box).to_numpy()]

        shortest_lines_to_way = shapely.shortest_line(np.full(len(ways_in_box), utm_point), ways_in_box)
        line_lengths = shapely.length(shortest_lines_to_way)

        nearest_way_ordering = np.argsort(line_lengths, kind="stable")[:max_candidates]
        nearest_endpoints = shapely.get_point(shortest_lines_to_way[nearest_way_ordering], -1)

        candidate_points = gpd.GeoSeries(nearest_endpoints, crs=utm_crs).to_crs(CRS)
        candidates[label] = (candidate_points, line_lengths[nearest_way_ordering])

    return candidates


def _nearest_walkable_points(points: gpd.GeoSeries, osm_path: Path) -> tuple[gpd.GeoSeries, np.ndarray]:
    """For each point in ``points``, return the nearest point on a walkable OSM way, and the distance to it in metres.

    Points with no walkable way within ``WALKABLE_SEARCH_RADIUS_DEG`` does not move.

    Args:
        points: Points in ``CRS``.
        osm_path: The OSM extract read for the ways.

    Returns:
        Tuple ``(nearest_points, distance_m)``: the nearest points in ``CRS``, in the order of ``points``, and the
        distance from each point to its nearest point.
    """
    point_x = points.x.to_numpy()
    point_y = points.y.to_numpy()
    radius = WALKABLE_SEARCH_RADIUS_DEG
    search_boxes = shapely.box(point_x - radius, point_y - radius, point_x + radius, point_y + radius)
    search_area = shapely.union_all(search_boxes)
    walkable_ways = gpd.GeoSeries(_read_walkable_ways(osm_path, search_area), crs=CRS)

    utm_crs = points.estimate_utm_crs()
    utm_points = points.to_crs(utm_crs)
    nearest_points = utm_points.copy()

    if len(walkable_ways) > 0:
        utm_ways = walkable_ways.to_crs(utm_crs)
        nearest_way_idx = utm_ways.sindex.nearest(utm_points, return_all=False)[1]
        nearest_ways = utm_ways.iloc[nearest_way_idx].to_numpy()
        shortest_lines = shapely.shortest_line(utm_points.to_numpy(), nearest_ways)
        nearest_points = gpd.GeoSeries(shapely.get_point(shortest_lines, -1), index=points.index, crs=utm_crs)

    distance_m = utm_points.distance(nearest_points).to_numpy()

    return nearest_points.to_crs(CRS), distance_m


def _route(
    r5py: ModuleType,
    network: Any,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    departure: dt.datetime,
) -> pd.DataFrame:
    """Return the travel times from ``origins`` to ``destinations`` on ``network``, as ``_pt_time_table`` does."""
    transport_modes = [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]
    matrix = r5py.TravelTimeMatrix(
        network,
        origins=origins,
        destinations=destinations,
        departure=departure,
        departure_time_window=PT_DEPARTURE_WINDOW,
        percentiles=[50],
        transport_modes=transport_modes,
        max_time=MAX_TRAVEL_TIME,
        speed_walking=WALK_SPEED_KMH,
    )

    return _pt_time_table(matrix)


def _unlinked_zone_ids(travel_times: pd.DataFrame) -> list[str]:
    """Return the zones that do not reach another zone, in index order."""
    source_ids = travel_times.index.get_level_values("source")
    target_ids = travel_times.index.get_level_values("target")
    other_pairs = travel_times[source_ids != target_ids]
    reaches_any = other_pairs["pt_reachable"].groupby(level="source", sort=False).any()

    return reaches_any[~reaches_any].index.tolist()


def _read_walkable_ways(osm_path: Path, search_area: shapely.Geometry) -> list[shapely.LineString]:
    """Return all walkable ``highway`` ways of an OSM extract that intersect ``search_area``, in ``CRS``."""
    shapely.prepare(search_area)
    highway_filter = osmium.filter.KeyFilter("highway")
    osm_entities = osmium.osm.NODE | osmium.osm.WAY
    way_iter = osmium.FileProcessor(str(osm_path), osm_entities).with_locations().with_filter(highway_filter)

    walkable_ways = []
    for way in way_iter:
        if not isinstance(way, osmium.osm.Way) or not _is_walkable(dict(way.tags)):
            continue

        way_coords = [(node.lon, node.lat) for node in way.nodes if node.location.valid()]
        if len(way_coords) < 2:
            continue

        way_line = shapely.LineString(way_coords)
        if shapely.intersects(search_area, way_line):
            walkable_ways.append(way_line)

    return walkable_ways


def _is_walkable(tags: dict[str, str]) -> bool:
    """Return whether the OSM tags of a ``highway`` way allow walking."""
    foot_value = tags.get("foot")
    if tags.get("highway") in NON_WALKABLE_HIGHWAYS or foot_value in NO_ACCESS_VALUES:
        return False

    return tags.get("access") not in NO_ACCESS_VALUES or foot_value in FOOT_ACCESS_VALUES


def _replace_pairs(travel_times: pd.DataFrame, rerouted_times: list[pd.DataFrame]) -> pd.DataFrame:
    """Return ``travel_times`` with the pairs of each of ``rerouted_times`` replaced, later frames taking precedence."""
    combined_times = pd.concat([travel_times, *rerouted_times])
    latest_times = combined_times[~combined_times.index.duplicated(keep="last")]

    return latest_times.reindex(travel_times.index)


def _restore_jpype_import_finder() -> None:
    """Restore the import finder of jpype instead of the r5py finder, which starts the JVM on every unknown import.

    This way, unknown imports after the JVM has shut down, such as an optional import in an exit hook, then fail as a
    missing module instead of raising an ``OSError`` from the JVM restart.
    """
    import jpype.imports

    for position, finder in enumerate(sys.meta_path):
        if isinstance(finder, jpype.imports._JImportLoader):
            sys.meta_path[position] = jpype.imports._JImportLoader()


def _pt_time_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the travel times of an R5 matrix of ``from_id, to_id, travel_time``, capped at ``MAX_TRAVEL_TIME``."""
    max_time_min = MAX_TRAVEL_TIME.total_seconds() / 60
    travel_time = matrix["travel_time"].to_numpy(dtype=np.float64)
    reachable = ~np.isnan(travel_time)
    pair_index = pd.MultiIndex.from_arrays([matrix["from_id"], matrix["to_id"]], names=EDGE_INDEX_NAMES)
    pair_columns = {"pt_time_min": np.where(reachable, travel_time, max_time_min), "pt_reachable": reachable}

    return pd.DataFrame(pair_columns, index=pair_index)
