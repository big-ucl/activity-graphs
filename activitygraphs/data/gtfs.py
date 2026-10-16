"""GTFS feed parser for building public-transport edge lists."""

from dataclasses import dataclass

import polars as pl
import polars.selectors as cs

from activitygraphs.base import PT_EDGE_LIST_SCHEMA, TRANSFER_EDGE_LIST_SCHEMA, Mode, PTNodeType
from activitygraphs.utils import check_schema

PARENT_STOP_ROUTE_ID = "parent"
DEFAULT_TRANSFER_TIME_MIN = 2


@dataclass(frozen=True)
class GTFSTables:
    """Parsed GTFS tables for a single feed (stops, stop times, trips, routes, agency, calendar, transfers)."""

    stops_df: pl.DataFrame
    stop_times_df: pl.LazyFrame
    trips_df: pl.LazyFrame
    routes_df: pl.DataFrame
    agency_df: pl.DataFrame
    calendar_df: pl.DataFrame
    calendar_dates_df: pl.DataFrame
    transfers_df: pl.DataFrame


ROUTE_TYPE_TO_MODE_MAP = {
    101: Mode.TRAIN,
    102: Mode.TRAIN,
    103: Mode.TRAIN,
    106: Mode.TRAIN,
    109: Mode.TRAIN,
    117: Mode.TRAIN,
    700: Mode.BUS,
    900: Mode.TRAMWAY,
    1000: Mode.BOAT,
}

AGENCY_NAME_MAP = {
    "Schweizerische Bundesbahnen SBB": "SBB",
    "Schweizerische Südostbahn (sob)": "SOB",
    "Transports Publics Genevois": "TPG",
    "BLS AG (bls)": "BLS",
    "Transports publics fribourgeois": "TPF",
    "Transports Publics de l'agglomération d'Annemasse": "TAC",
}

MODE_TO_SHORT_NAME_MAP = {
    Mode.TRAIN: "Train : ",
    Mode.TRAMWAY: "Tram  : ",
    Mode.BUS: "Bus  : ",
    Mode.BOAT: "Boat : ",
}


SAMPLE_DATE = pl.date(2022, 5, 18)
SAMPLE_WEEKDAY = "wednesday"
SAMPLE_DAY_START = pl.time(6, 0, 0)
SAMPLE_DAY_END = pl.time(21, 0, 0)


def build_pt_layer_edges(
    locations_df: pl.DataFrame,
    gtfs: GTFSTables,
    pt_node_type: PTNodeType,
    drop_null_headways: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build PT and transfer edge DataFrames from a GTFS feed for the given locations.

    Args:
        locations_df: Active location IDs; only stops within this set are included.
        gtfs: Parsed GTFS tables.
        pt_node_type: Whether to create one node per route (``ONE_PER_ROUTE``) or per stop (``ONE_PER_STOP``).
        drop_null_headways: If True, remove PT edges with no computed headway.

    Returns:
        Tuple ``(pt_edge_df, transfer_edge_df)`` conforming to ``PT_EDGE_LIST_SCHEMA``
        and ``TRANSFER_EDGE_LIST_SCHEMA`` respectively.
    """
    edge_ids = (
        ["route_id", "orig_loc_id", "dest_loc_id"]
        if pt_node_type == PTNodeType.ONE_PER_ROUTE
        else ["orig_loc_id", "dest_loc_id"]
    )

    active_stop_times = filter_active_stop_times(locations_df, gtfs)
    pt_trips = create_pt_trip_df(active_stop_times)
    headways_df = compute_avg_headways(pt_trips, gtfs, edge_ids)

    pt_edge_df = pt_trips.group_by(edge_ids).agg(
        first_departure_time=pl.col("orig_departure_time").min(),
        last_departure_time=pl.col("orig_departure_time").max(),
        avg_dwell_time_min=pl.col("orig_dwell_time_min").mean(),
        travel_time_min=pl.col("travel_time_min").mean(),
    )

    if pt_node_type == PTNodeType.ONE_PER_STOP:
        pt_edge_df = pt_edge_df.with_columns(route_id=pl.lit(PARENT_STOP_ROUTE_ID))

    pt_edge_df = pt_edge_df.join(headways_df, on=edge_ids, how="left").collect()
    pt_edge_df = add_route_attributes_to_edges(pt_edge_df, gtfs, pt_node_type)
    pt_edge_df = pt_edge_df.select(PT_EDGE_LIST_SCHEMA.keys())

    if drop_null_headways:
        pt_edge_df = pt_edge_df.drop_nulls("avg_headway_min")

    transfer_edge_df = create_transfer_edges(pt_edge_df, locations_df, gtfs)

    return check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA), check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)


def filter_active_stop_times(locations_df: pl.DataFrame, gtfs: GTFSTables) -> pl.LazyFrame:
    """Filter GTFS stop times to trips that visit at least two active locations."""
    # Filter out all stops not in the list of active locations
    # Only keep trips that connect at least 2 active locations
    return (
        gtfs.stop_times_df
        .filter(pl.col("loc_id").is_in(locations_df["loc_id"].implode()))
        .filter((pl.len() >= 2).over("trip_id"))
        .with_columns(pl.col("stop_sequence").rank("dense").over("trip_id", order_by="stop_sequence"))
        .join(gtfs.trips_df.select("route_id", "service_id", "trip_id"), on="trip_id")
    )


def create_pt_trip_df(active_stop_times: pl.LazyFrame) -> pl.LazyFrame:
    """Join consecutive stop-time pairs into origin-destination trip segments with dwell and travel times."""
    join_cols = ["trip_id", "service_id", "route_id"]
    departures = active_stop_times.select(*join_cols, pl.exclude(join_cols).name.prefix("orig_"))
    arrivals = active_stop_times.select(*join_cols, pl.exclude(join_cols).name.prefix("dest_"))

    return (
        departures
        .join(
            arrivals,
            left_on=[*join_cols, pl.col("orig_stop_sequence")],
            right_on=[*join_cols, pl.col("dest_stop_sequence") - 1],
        )
        .drop("trip_id_right", "service_id_right", "route_id_right")
        .with_columns(cs.ends_with("time").str.to_time("%T", strict=False))
        .drop_nulls()
        .with_columns(
            orig_dwell_time_min=(pl.col("orig_departure_time") - pl.col("orig_arrival_time")).dt.total_minutes(
                fractional=True
            ),
            travel_time_min=(pl.col("dest_arrival_time") - pl.col("orig_departure_time")).dt.total_minutes(
                fractional=True
            ),
        )
    )


def compute_avg_headways(pt_trips: pl.LazyFrame, gtfs: GTFSTables, edge_id: list[str]) -> pl.LazyFrame:
    """Compute daily trip count and average headway for each edge on the sample weekday (Wednesday 2022-05-18)."""
    exceptions = gtfs.calendar_dates_df.filter(date=SAMPLE_DATE)
    services_added = exceptions.filter(exception_type=1).select("service_id")
    services_removed = exceptions.filter(exception_type=2).select("service_id")

    sample_services = (
        gtfs.calendar_df
        .filter(pl.col(SAMPLE_WEEKDAY) == 1)
        .select("service_id")
        .join(services_added, on="service_id", how="left")
        .join(services_removed, on="service_id", how="anti")
    )

    return (
        pt_trips
        .join(sample_services.lazy(), on="service_id")
        .filter((SAMPLE_DAY_START <= pl.col("orig_departure_time")) & (pl.col("orig_departure_time") <= SAMPLE_DAY_END))
        .group_by(edge_id)
        .agg(
            daily_trip_count=pl.len(),
            avg_headway_min=pl
            .col("orig_departure_time")
            .sort()
            .diff(null_behavior="drop")
            .dt.total_minutes(fractional=True)
            .mean(),
        )
    )


def add_route_attributes_to_edges(edge_df: pl.DataFrame, gtfs: GTFSTables, pt_node_type: PTNodeType) -> pl.DataFrame:
    """Add route mode and name onto the PT edge DataFrame."""
    if pt_node_type == PTNodeType.ONE_PER_STOP:
        return edge_df.with_columns(
            route_mode=pl.lit(Mode.UNKNOWN).cast(pl.Categorical), route_name=pl.lit(PARENT_STOP_ROUTE_ID)
        )

    agencies = gtfs.agency_df.select("agency_id", "agency_name")

    route_mode = pl.col("route_type").replace_strict(
        ROUTE_TYPE_TO_MODE_MAP, default=Mode.UNKNOWN, return_dtype=pl.Categorical
    )
    route_mode_desc = pl.col("route_mode").cast(pl.String).replace(MODE_TO_SHORT_NAME_MAP)
    agency_name = pl.lit("  (") + pl.col("agency_name").replace(AGENCY_NAME_MAP) + pl.lit(")")
    route_name = pl.col("route_mode_desc") + pl.col("route_short_name") + agency_name

    active_routes_df = (
        gtfs.routes_df
        .join(agencies, on="agency_id")
        .with_columns(route_mode=route_mode)
        .with_columns(route_mode_desc=route_mode_desc)
        .select("route_id", "route_mode", route_name=route_name)
    )

    return edge_df.join(active_routes_df, on="route_id")


def create_transfer_edges(pt_edge_df: pl.DataFrame, locations_df: pl.DataFrame, gtfs: GTFSTables) -> pl.DataFrame:
    """Build transfer edges within PT stops (route-to-route) and between PT stops (transfer-node to transfer-node) from GTFS transfer times."""
    transfer_route_id = PARENT_STOP_ROUTE_ID

    # Include only stops appearing in the edge list
    loc_ids = locations_df.select("loc_id")
    stops = gtfs.stops_df.join(loc_ids, on="loc_id")

    # Compute average transfer time in minutes between loc_ids
    gtfs_transfers = (
        gtfs.transfers_df
        .join(stops.select("stop_id", orig_loc_id="loc_id"), left_on="from_stop_id", right_on="stop_id")
        .join(stops.select("stop_id", dest_loc_id="loc_id"), left_on="to_stop_id", right_on="stop_id")
        .group_by("orig_loc_id", "dest_loc_id")
        .agg(pl.col("min_transfer_time").unique())
        .with_columns(travel_time_min=pl.col("min_transfer_time").list.mean() / 60)
        .select("orig_loc_id", "dest_loc_id", "travel_time_min")
    )

    # For every (route_id, loc_id) pair, add an incoming edge from the pair to the central (transfer_route_id, loc_id) node
    # The incoming edge has a transfer time defined in the GTFS or the default transfer time if not defined
    pt_nodes_df = pl.concat([
        pt_edge_df.select("route_id", loc_id="orig_loc_id"),
        pt_edge_df.select("route_id", loc_id="dest_loc_id"),
    ]).unique()

    incoming_transfers = pt_nodes_df.select(
        orig_loc_id="loc_id", orig_route_id="route_id", dest_loc_id="loc_id", dest_route_id=pl.lit(transfer_route_id)
    )
    incoming_transfers = incoming_transfers.join(
        gtfs_transfers, on=["orig_loc_id", "dest_loc_id"], how="left"
    ).with_columns(pl.col("travel_time_min").fill_null(DEFAULT_TRANSFER_TIME_MIN))

    # Similarly, an outgoing edge from the central (transfer_route_id, loc_id) node to the (route_id, loc_id) pair, with no travel time
    outgoing_transfers = pt_nodes_df.select(
        orig_loc_id="loc_id",
        orig_route_id=pl.lit(transfer_route_id),
        dest_loc_id="loc_id",
        dest_route_id="route_id",
        travel_time_min=0.0,
    )

    # Add the GTFS transfers between different loc_ids
    inter_loc_transfers = (
        gtfs_transfers
        .filter(pl.col("orig_loc_id") != pl.col("dest_loc_id"))
        .with_columns(orig_route_id=pl.lit(transfer_route_id), dest_route_id=pl.lit(transfer_route_id))
        .select(incoming_transfers.columns)
    )

    return pl.concat([inter_loc_transfers, incoming_transfers, outgoing_transfers])
