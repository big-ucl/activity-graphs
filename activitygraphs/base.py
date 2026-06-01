"""Enums (Mode, Purpose) and Polars/GeoPandas schemas shared across all modules."""

from enum import StrEnum

import polars as pl


class PTNodeType(StrEnum):
    """Type of node for PT graph nodes: one node per route or one per stop."""

    ONE_PER_ROUTE = "ONE_PER_ROUTE"
    ONE_PER_STOP = "ONE_PER_STOP"


class Mode(StrEnum):
    """Travel modes used as categorical values in journey legs."""

    OTHER = "mode_other"
    UNKNOWN = "mode_unknown"
    BOAT = "mode_boat"
    BUS = "mode_bus"
    COACH = "mode_coach"
    WALK = "mode_walk"
    CYCLE = "mode_cycle"
    MOTORCYCLE = "mode_motorcycle"
    TAXI = "mode_taxi"
    TRAIN = "mode_train"
    TRAMWAY = "mode_tramway"
    VEH_PASS = "mode_vehicle_passenger"
    CAR = "mode_car"


class Purpose(StrEnum):
    """Trip purposes used as categorical values in journey departure/arrival records."""

    OTHER = "purp_other"
    UNKNOWN = "purp_unknown"
    HOME = "purp_home"
    WORK_MAIN = "purp_work"
    WORK_OTHER = "purp_work_other"
    STUDY = "purp_study"
    VISIT = "purp_visit"
    ESCORT = "purp_escort"
    PERSONAL = "purp_personal"
    SHOP = "purp_shopping"
    ENTERTAINMENT = "purp_entertainment"
    LEISURE_OTHER = "purp_leisure_other"
    LONG_DISTANCE_TRIP = "purp_long_distance_trip"


LOCATIONS_SCHEMA = {
    "loc_id": "object",
    "loc_name": "object",
    "type": "object",
    "lon": "float64",
    "lat": "float64",
    "geometry": "geometry",
}
LOCATIONS_COLUMNS = list(LOCATIONS_SCHEMA.keys())

USER_JOURNEY_SCHEMA = pl.Schema({
    "user_id": pl.String,
    "journey_id": pl.String,
    "leg_id": pl.Int8,
    "leg_mode": pl.Categorical(),
    "leg_line": pl.String,
    "duration": pl.Duration(),
    "dep_day": pl.Date,
    "dep_time": pl.Time,
    "dep_purpose": pl.Categorical(),
    "dep_loc_id": pl.String,
    "arr_loc_id": pl.String,
    "arr_purpose": pl.Categorical(),
})

USER_SCHEMA = pl.Schema({
    "user_id": pl.String,
    "home_loc_id": pl.String,
    "hh_num_adults": pl.UInt32,
    "hh_num_children": pl.UInt32,
})

EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "dest_loc_id": pl.String,
    "travel_time_min": pl.Float64,
})

WALK_EDGE_LIST_SCHEMA = EDGE_LIST_SCHEMA
LINK_EDGE_LIST_SCHEMA = EDGE_LIST_SCHEMA

PT_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "route_id": pl.String,
    "dest_loc_id": pl.String,
    "first_departure_time": pl.Time,
    "last_departure_time": pl.Time,
    "avg_dwell_time_min": pl.Float64,
    "travel_time_min": pl.Float64,
    "daily_trip_count": pl.UInt32,
    "avg_headway_min": pl.Float64,
    "route_mode": pl.Categorical(),
    "route_name": pl.String,
})

TRANSFER_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "orig_route_id": pl.String,
    "dest_loc_id": pl.String,
    "dest_route_id": pl.String,
    "travel_time_min": pl.Float64,
})
CRS = "EPSG:4326"
