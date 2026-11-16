"""Rail edges: the zone pairs joined by a rail or metro trip, with the in-vehicle time and the number of trips."""

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pyproj import CRS as ProjCRS

from activitygraphs.graph.access import RAIL_MODES, rail_stations
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.surveys.gtfs import GTFSTables

RAIL_EDGE_SCHEMA = TensorSchema(
    "rail_edges",
    (
        ColumnSpec("rail_time_s", ColumnKind.COUNT),
        ColumnSpec("rail_trips", ColumnKind.COUNT),
    ),
)
"""Columns of the rail edges built by ``build_rail_edges``, in tensor column order."""

EDGE_INDEX_NAMES = ["source", "target"]
"""Names of the two ``loc_id`` levels of the index of an edge table."""


def build_rail_edges(nodes_gdf: gpd.GeoDataFrame, gtfs: GTFSTables) -> pd.DataFrame:
    """Return the pairs of zones connected by a rail or metro trip on the service date.

    Two zones are connected when one rail or metro trip calls at a station in each zone. A station belongs to a zone if
    it is located inside the zone's polygon and stations are mapped to the GTFS parent stations if provided. Each edge
    includes two features:
     - ``rail_time_s``: the minimum `route` time between the two zones.
     - ``rail_trips``: the total number of rail trips connecting the two zones, over all routes.

    The `trip` time is ``arrival_time - departure_time`` between the closest stops in each zone, and the `route` time is
    the median `trip` time over all trips on that route.

    Args:
        nodes_gdf: Zones indexed by ``loc_id``, in a projected CRS, with the zone polygon in ``original_geometry``.
        gtfs: The GTFS feeds of the dataset.

    Returns:
        One row per zone pair, indexed by the (``source``, ``target``) ``loc_id`` of the two zones, the source first in
        node order, with columns ``rail_time_s`` and ``rail_trips``.

    Raises:
        ValueError: ``nodes_gdf`` has no CRS.
    """
    zone_crs = nodes_gdf.crs

    if zone_crs is None:
        raise ValueError("The zones have no CRS.")

    station_zones = _station_zones(nodes_gdf, zone_crs, gtfs)
    calls = _rail_calls(gtfs, station_zones)

    trip_pairs = (
        calls
        .join(calls, on="trip_id", suffix="_later")
        .filter(
            (pl.col("stop_sequence") < pl.col("stop_sequence_later")) & (pl.col("zone_row") != pl.col("zone_row_later"))
        )
        .select(
            "trip_id",
            "route_id",
            source_row=pl.min_horizontal("zone_row", "zone_row_later"),
            target_row=pl.max_horizontal("zone_row", "zone_row_later"),
            time_s=pl.col("arrival_s_later") - pl.col("departure_s"),
        )
    )
    edges = (
        trip_pairs
        .group_by("trip_id", "route_id", "source_row", "target_row")
        .agg(pl.col("time_s").min())
        .group_by("route_id", "source_row", "target_row")
        .agg(pl.col("time_s").median(), n_trips=pl.len())
        .group_by("source_row", "target_row")
        .agg(rail_time_s=pl.col("time_s").min(), rail_trips=pl.col("n_trips").sum())
        .sort("source_row", "target_row")
    )

    loc_ids = nodes_gdf.index.to_numpy()
    source_ids = loc_ids[edges["source_row"].to_numpy()]
    target_ids = loc_ids[edges["target_row"].to_numpy()]
    pair_index = pd.MultiIndex.from_arrays([source_ids, target_ids], names=EDGE_INDEX_NAMES)
    edge_columns = {"rail_time_s": edges["rail_time_s"].to_numpy(), "rail_trips": edges["rail_trips"].to_numpy()}

    return pd.DataFrame(edge_columns, index=pair_index)


def _station_zones(nodes_gdf: gpd.GeoDataFrame, zone_crs: ProjCRS, gtfs: GTFSTables) -> pl.DataFrame:
    """Return the row in ``nodes_gdf`` of the zone that contains each rail or metro station, as ``station_id, zone_row``."""
    polygon_geometry = gpd.GeoSeries(nodes_gdf["original_geometry"].to_numpy(), crs=zone_crs)
    zone_polygons = gpd.GeoDataFrame({"zone_row": np.arange(len(nodes_gdf))}, geometry=polygon_geometry)

    stations = rail_stations(gtfs).to_crs(zone_crs)
    station_zones = stations.sjoin(zone_polygons, predicate="within")

    return pl.DataFrame({
        "station_id": station_zones["station_id"].to_list(),
        "zone_row": station_zones["zone_row"].to_numpy(),
    })


def _rail_calls(gtfs: GTFSTables, station_zones: pl.DataFrame) -> pl.DataFrame:
    """Return the calls (stoptime) of the rail and metro trips, on the service date at the stations inside a zone.

    Calls with a null/empty time get assigned a time interpolated linearly, in stop sequence, between the two calls that
    have a non-null time.

    Returns:
        Frame of ``trip_id, route_id, stop_sequence, zone_row, departure_s, arrival_s``, one row per call.
    """
    rail_routes = gtfs.routes_df.filter(pl.col("mode").is_in(RAIL_MODES)).select("route_id")
    rail_trips = gtfs.trips_on_service_date().join(rail_routes.lazy(), on="route_id").select("trip_id", "route_id")
    stop_stations = gtfs.stops_df.select("stop_id", station_id=pl.coalesce("parent_station", "stop_id"))
    departure_s = pl.coalesce("departure_time", "arrival_time").cast(pl.Float64).interpolate().over("trip_id")
    arrival_s = pl.coalesce("arrival_time", "departure_time").cast(pl.Float64).interpolate().over("trip_id")

    return (
        gtfs.stop_times_df
        .join(rail_trips, on="trip_id")
        .sort("trip_id", "stop_sequence")
        .with_columns(departure_s=departure_s, arrival_s=arrival_s)
        .join(stop_stations.lazy(), on="stop_id")
        .join(station_zones.lazy(), on="station_id")
        .select("trip_id", "route_id", "stop_sequence", "zone_row", "departure_s", "arrival_s")
        .collect()
    )
