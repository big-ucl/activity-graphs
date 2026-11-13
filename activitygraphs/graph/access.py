"""Accessibility features of the zones: rail and metro stations, morning-peak departures and motorway junctions."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import polars as pl
from scipy.spatial import KDTree

from activitygraphs.base import CRS
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.surveys.gtfs import GTFSTables, PTMode

ACCESS_SCHEMA = TensorSchema(
    "access",
    (
        ColumnSpec("has_rail_station", ColumnKind.BINARY),
        ColumnSpec("rail_station_distance_m", ColumnKind.COUNT),
        ColumnSpec("peak_departures_per_hour", ColumnKind.COUNT),
        ColumnSpec("motorway_junction_distance_m", ColumnKind.COUNT),
    ),
)
"""Columns added to the zones by ``add_access_features``, in tensor column order."""

RAIL_MODES = [PTMode.RAIL.value, PTMode.METRO.value]

# The morning peak, in seconds after midnight of the service day, start included and end excluded.
PEAK_START_S = 7 * 3600
PEAK_END_S = 9 * 3600

MOTORWAY_JUNCTION_TAG = ("highway", "motorway_junction")


def add_access_features(nodes_gdf: gpd.GeoDataFrame, gtfs: GTFSTables, junctions: gpd.GeoSeries) -> gpd.GeoDataFrame:
    """Return ``nodes_gdf`` with the ``ACCESS_SCHEMA`` columns.

    A zone has a rail station when the zone polygon contains a rail or metro station that is served on the service date.
    Both distances are measured from the zone centroid. The peak departures count the departures of every mode from
    the stops inside the zone polygon between ``PEAK_START_S`` and ``PEAK_END_S``, per hour.

    Args:
        nodes_gdf: Zones in a projected CRS in metres, with the zone point (centroid) as the geometry and the zone
            polygon in ``original_geometry``.
        gtfs: The GTFS feeds of the dataset.
        junctions: Motorway junction points, in any CRS.

    Returns:
        A copy of ``nodes_gdf`` with the ``ACCESS_SCHEMA`` columns added.

    Raises:
        ValueError: ``nodes_gdf`` has no CRS.
    """
    zone_crs = nodes_gdf.crs

    if zone_crs is None:
        raise ValueError("The zones have no CRS.")

    n_zones = len(nodes_gdf)
    zone_points = nodes_gdf.geometry.get_coordinates().to_numpy()
    polygon_geometry = gpd.GeoSeries(nodes_gdf["original_geometry"].to_numpy(), crs=zone_crs)
    zone_polygons = gpd.GeoDataFrame({"zone_row": np.arange(n_zones)}, geometry=polygon_geometry)

    stations = rail_stations(gtfs).to_crs(zone_crs)
    station_zones = gpd.GeoDataFrame(geometry=stations.geometry).sjoin(zone_polygons, predicate="within")
    stations_per_zone = np.bincount(station_zones["zone_row"].to_numpy(), minlength=n_zones)

    stop_departures = peak_departures(gtfs)
    stop_points = gpd.points_from_xy(stop_departures["stop_lon"], stop_departures["stop_lat"], crs=CRS)
    stop_frame = gpd.GeoDataFrame({"n_departures": stop_departures["n_departures"]}, geometry=stop_points)
    stop_zones = stop_frame.to_crs(zone_crs).sjoin(zone_polygons, predicate="within")
    stop_zone_rows = stop_zones["zone_row"].to_numpy()
    stop_zone_departures = stop_zones["n_departures"].to_numpy(dtype=np.float64)
    departures_per_zone = np.bincount(stop_zone_rows, weights=stop_zone_departures, minlength=n_zones)
    peak_hours = (PEAK_END_S - PEAK_START_S) / 3600

    station_points = stations.geometry.get_coordinates().to_numpy()
    junction_points = junctions.to_crs(zone_crs).get_coordinates().to_numpy()

    with_access = nodes_gdf.copy()
    with_access["has_rail_station"] = (stations_per_zone > 0).astype(np.int64)
    with_access["rail_station_distance_m"] = _nearest_distance(zone_points, station_points)
    with_access["peak_departures_per_hour"] = departures_per_zone / peak_hours
    with_access["motorway_junction_distance_m"] = _nearest_distance(zone_points, junction_points)

    return with_access


def rail_stations(gtfs: GTFSTables) -> gpd.GeoDataFrame:
    """Return the rail and metro stations served on the service date, as points in ``CRS``.

    A stop counts as its parent station where the feed gives one.

    Returns:
        One row per station, with a ``station_id`` column, sorted by ``station_id``.
    """
    rail_routes = gtfs.routes_df.filter(pl.col("mode").is_in(RAIL_MODES)).select("route_id")
    rail_trips = gtfs.trips_on_service_date().join(rail_routes.lazy(), on="route_id").select("trip_id")
    stop_stations = gtfs.stops_df.select("stop_id", station_id=pl.coalesce("parent_station", "stop_id"))
    served_stops = gtfs.stop_times_df.join(rail_trips, on="trip_id").select("stop_id").unique().collect()

    station_coords = gtfs.stops_df.select(station_id="stop_id", lon="stop_lon", lat="stop_lat")
    stations = (
        served_stops
        .join(stop_stations, on="stop_id")
        .select("station_id")
        .unique()
        .join(station_coords, on="station_id")
        .sort("station_id")
    )
    station_points = gpd.points_from_xy(stations["lon"], stations["lat"], crs=CRS)

    return gpd.GeoDataFrame({"station_id": stations["station_id"].to_list()}, geometry=station_points)


def peak_departures(gtfs: GTFSTables) -> pl.DataFrame:
    """Return the number of departures from each stop between ``PEAK_START_S`` and ``PEAK_END_S`` on the service date.

    A call with neither a departure nor an arrival time takes a time interpolated along its trip, linearly in stop
    sequence between the calls that have one. The last call of a trip is not a departure.

    Returns:
        Frame of ``stop_id, stop_lon, stop_lat, n_departures``, one row per stop with at least one departure.
    """
    service_trips = gtfs.trips_on_service_date().select("trip_id")
    call_time = pl.coalesce("departure_time", "arrival_time").cast(pl.Float64)
    interpolated_time = pl.col("call_time").interpolate().over("trip_id")
    is_last_call = pl.col("stop_sequence") == pl.col("stop_sequence").max().over("trip_id")
    in_peak = pl.col("call_time").is_between(PEAK_START_S, PEAK_END_S, closed="left")

    stop_departures = (
        gtfs.stop_times_df
        .join(service_trips, on="trip_id")
        .select("trip_id", "stop_id", "stop_sequence", call_time=call_time)
        .sort("trip_id", "stop_sequence")
        .with_columns(call_time=interpolated_time, is_last_call=is_last_call)
        .filter(~pl.col("is_last_call") & in_peak)
        .group_by("stop_id")
        .agg(n_departures=pl.len())
        .collect()
    )
    stop_coords = gtfs.stops_df.select("stop_id", "stop_lon", "stop_lat")

    return stop_departures.join(stop_coords, on="stop_id").select("stop_id", "stop_lon", "stop_lat", "n_departures")


def read_motorway_junctions(osm_path: Path) -> gpd.GeoSeries:
    """Return the ``highway=motorway_junction`` nodes of an OSM extract, as points in ``CRS``.

    Raises:
        ValueError: The extract does not contain motorway junctions.
    """
    tag_filter = osmium.filter.TagFilter(MOTORWAY_JUNCTION_TAG)

    junction_iter = osmium.FileProcessor(str(osm_path), osmium.osm.NODE).with_filter(tag_filter)
    junction_nodes = (node for node in junction_iter if isinstance(node, osmium.osm.Node))
    junction_coords = [(node.location.lon, node.location.lat) for node in junction_nodes]

    if not junction_coords:
        raise ValueError(f"The OSM extract {osm_path} does not contain motorway junctions.")

    lons, lats = zip(*junction_coords, strict=True)
    return gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=CRS)


def _nearest_distance(origins: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return the distance from each of ``origins`` ``[N, 2]`` to the nearest of ``targets`` ``[M, 2]``, same units.

    Raises:
        ValueError: ``targets`` is empty.
    """
    if len(targets) == 0:
        raise ValueError("No target point to measure a distance to.")

    distances, _ = KDTree(targets).query(origins)
    return np.asarray(distances, dtype=np.float64)
