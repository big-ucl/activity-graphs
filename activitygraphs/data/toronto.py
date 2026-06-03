"""Toronto THATS survey loader and NetworkData subclass."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import polars.selectors as cs

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA, Mode, Purpose
from activitygraphs.config import TorontoDataConfig
from activitygraphs.network import NA, NetworkData, build_special_locations
from activitygraphs.utils import (
    DataFrameStore,
    add_lon_lat_from_centroid,
    check_schema,
    get_project_root,
    invert_mapping,
)

TORONTO_CMA = 535

MODE_MAP = {
    "AIR_OR_HSR": Mode.OTHER,
    "BICYCLING": Mode.CYCLE,
    "BUS": Mode.BUS,
    "CAR": Mode.CAR,
    "FERRY": Mode.BOAT,
    "LIGHT_RAIL": Mode.TRAMWAY,
    "SUBWAY": Mode.TRAIN,
    "TRAIN": Mode.TRAIN,
    "TRAM": Mode.TRAMWAY,
    "TROLLEYBUS": Mode.BUS,
    "UNKNOWN": Mode.UNKNOWN,
    "WALKING": Mode.WALK,
}

MANUAL_MODE_MAP = {
    "walk": Mode.WALK,
    "car-park-off-street-res": Mode.CAR,
    "car-park-off-street-non-res": Mode.CAR,
    "transit": Mode.BUS,
    "car-passenger": Mode.VEH_PASS,
    "bike": Mode.CYCLE,
    "car-park-on-street-res": Mode.CAR,
    "car-park-on-street-non-res": Mode.CAR,
    "train": Mode.TRAIN,
    "taxi": Mode.TAXI,
    "train-transit": Mode.TRAIN,
    "golf_cart": Mode.OTHER,
    "operate_heavy equipment": Mode.OTHER,
    "work/tractor": Mode.OTHER,
    "run": Mode.WALK,
    "boat": Mode.BOAT,
    "shunting": Mode.CAR,
    "auto": Mode.CAR,
    "driving_company vehicle for work": Mode.CAR,
    "motorcycle": Mode.MOTORCYCLE,
    '"auto': Mode.CAR,
    "auto_(drive) parked in driveway": Mode.CAR,
    "airplane": Mode.OTHER,
    "auto_(work related)": Mode.CAR,
    "car": Mode.CAR,
    "canoe": Mode.BOAT,
    "subway": Mode.TRAIN,
    "not_aplicable": Mode.UNKNOWN,
    "transit_+ walk": Mode.BUS,
    "walk_and transit": Mode.BUS,
    "auto_(work)": Mode.CAR,
    "delivery": Mode.CAR,
    '"walking': Mode.WALK,
    "ferry": Mode.BOAT,
    "sail": Mode.BOAT,
}

PURPOSE_MAP = {
    str(Purpose.UNKNOWN): Purpose.UNKNOWN,
    "home": Purpose.HOME,
    "entertainment-visit": Purpose.ENTERTAINMENT,
    "excercice-recreation": Purpose.LEISURE_OTHER,
    "work": Purpose.WORK_MAIN,
    "shopping-grocerie": Purpose.SHOP,
    "pick-drop": Purpose.ESCORT,
    "appointment-visit": Purpose.PERSONAL,
    "personal-business": Purpose.PERSONAL,
    "school": Purpose.STUDY,
    "dog_walk": Purpose.LEISURE_OTHER,
    "church": Purpose.LEISURE_OTHER,
    "shopping": Purpose.SHOP,
    "golf": Purpose.LEISURE_OTHER,
    "walk_dog": Purpose.LEISURE_OTHER,
    "shunting": Purpose.ESCORT,
    "return_home": Purpose.HOME,
    "volunteering": Purpose.LEISURE_OTHER,
    "vacation": Purpose.LONG_DISTANCE_TRIP,
    "volunteer": Purpose.LEISURE_OTHER,
    "visit_family": Purpose.VISIT,
    "lunch": Purpose.ENTERTAINMENT,
    "travel": Purpose.LONG_DISTANCE_TRIP,
    "religious": Purpose.LEISURE_OTHER,
    "going_home": Purpose.HOME,
    "parcel_deliveries in brampton": Purpose.WORK_OTHER,
    "library": Purpose.LEISURE_OTHER,
    "cat_sitting for a friend": Purpose.VISIT,
    "coffee": Purpose.ENTERTAINMENT,
    "family_visit": Purpose.VISIT,
    "walk": Purpose.LEISURE_OTHER,
    "traveling": Purpose.LONG_DISTANCE_TRIP,
    # More can be added
}

ACTIVITY_MAP = invert_mapping({
    Purpose.OTHER: [41],
    Purpose.UNKNOWN: [43],
    Purpose.HOME: [*range(1, 14)],
    Purpose.WORK_MAIN: [14],
    Purpose.WORK_OTHER: [],
    Purpose.STUDY: [15],
    Purpose.VISIT: [17, 40],
    Purpose.ESCORT: [26],
    Purpose.PERSONAL: [*range(36, 40)],
    Purpose.SHOP: [*range(29, 36)],
    Purpose.ENTERTAINMENT: [16, 18, 22, 24],
    Purpose.LEISURE_OTHER: [19, 20, 21, 23, 25, 27, 28],
    Purpose.LONG_DISTANCE_TRIP: [42],
})


@dataclass(frozen=True)
class TorontoInputs:
    """Parsed raw inputs for the Toronto THATS survey (journeys, persons, households, activities, boundaries)."""

    raw_journeys_df: pl.DataFrame
    raw_person_df: pl.DataFrame
    raw_household_df: pl.DataFrame
    raw_activities_df: pl.LazyFrame

    metropolitan_areas_gdf: gpd.GeoDataFrame
    census_tracts_gdf: gpd.GeoDataFrame
    dissemination_areas_gdf: gpd.GeoDataFrame


class TorontoData(NetworkData, DataFrameStore):
    """Toronto THATS ``NetworkData`` subclass with caching via ``DataFrameStore``."""

    def __init__(
        self,
        inputs: TorontoInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        activities_df: pl.DataFrame,
        users_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        super().__init__(user_journeys_df, locations_gdf, filters)
        self.inputs = inputs
        self.activities_df = activities_df
        self._users_df = users_df

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        return self.users_df.select("user_id", loc_id="home_loc_id")

    @cached_property
    def users_df(self) -> pl.DataFrame:
        return self._filter_loc_types(self._users_df, "home_loc_id").sort("user_id")

    @cached_property
    def user_ids(self) -> pl.Series:
        return self.users_df["user_id"].sort()

    @cached_property
    def num_obs_days_per_user(self) -> pl.DataFrame:
        """Per-user observed-day count t_i (0-7) from the activity diary."""
        return (
            self.activities_df
            .group_by("person_id")
            .agg(n_days=pl.col("act_date").n_unique().cast(pl.Int32))
            .rename({"person_id": "user_id"})
        )

    def _copy(self, filters: list[str] | None = None):
        return TorontoData(
            self.inputs, self._locations_gdf, self._user_journeys_df, self.activities_df, self.users_df, filters
        )

    @classmethod
    def load(cls, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None) -> "TorontoData":
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        toronto_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)
            activities_df = pl.read_parquet(data_dir / "activities_df.parquet")
            users_df = pl.read_parquet(data_dir / "users_df.parquet")

            return cls(toronto_inputs, locations_gdf, user_journeys_df, activities_df, users_df)
        else:
            data = build_toronto_data(toronto_inputs)
            data.save(cfg, project_root, name)

            return data

    def save(self, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None):
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self.locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self.user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")
        self.activities_df.write_parquet(data_dir / "activities_df.parquet")
        self.users_df.write_parquet(data_dir / "users_df.parquet")


def load_files(cfg: TorontoDataConfig, project_root: Path | None = None) -> TorontoInputs:
    """Read all raw Toronto files (journeys, persons, households, activities, boundaries) into a ``TorontoInputs`` container."""
    project_root = get_project_root(project_root)

    raw_data_dir = project_root / cfg.paths.raw
    raw_journeys_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_journeys)
    raw_persons_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_person)
    raw_household_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_household)
    raw_activities_df = pl.scan_parquet(raw_data_dir / cfg.inputs.raw_activities)

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    boundary_cma = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.metropolitan_areas)
    boundary_ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)
    boundary_da = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.dissemination_areas)

    return TorontoInputs(
        raw_journeys_df, raw_persons_df, raw_household_df, raw_activities_df, boundary_cma, boundary_ct, boundary_da
    )


def build_toronto_data(inputs: TorontoInputs) -> TorontoData:
    """Parse raw Toronto inputs into a standardised ``TorontoData`` instance."""
    locations_gdf = build_toronto_locations(inputs)
    user_journeys_df = build_toronto_journeys(inputs, locations_gdf)
    activities_df = build_toronto_activities(inputs, user_journeys_df)
    users_df = build_toronto_users(inputs, locations_gdf, user_journeys_df)

    return TorontoData(inputs, locations_gdf, user_journeys_df, activities_df, users_df)


# =========================================
# Locations
# =========================================


def build_toronto_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    """Build the Toronto locations GeoDataFrame (census tracts and NA)."""
    special_locations = build_special_locations()
    subsector_locations = build_subsector_locations(inputs)
    return gpd.GeoDataFrame(
        pd.concat(
            [
                special_locations,
                subsector_locations,
            ],
            ignore_index=True,
        ),
        crs=CRS,
    )


def build_subsector_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    """Extract census-tract "subsector" locations within the Toronto CMA and return them as subsector locations."""
    toronto_cma = inputs.metropolitan_areas_gdf.query(f"CMAUID == '{TORONTO_CMA}'")
    utm_crs = toronto_cma.estimate_utm_crs()

    boundaries_gdf = inputs.census_tracts_gdf
    boundaries_gdf = boundaries_gdf.to_crs(utm_crs).sjoin(toronto_cma.to_crs(utm_crs))
    boundaries_gdf = boundaries_gdf.drop(columns=["index_right"]).reset_index(drop=True).to_crs(CRS)

    boundaries_gdf = boundaries_gdf.rename(columns={"CTUID": "loc_id", "CTNAME": "loc_name"})
    boundaries_gdf["type"] = "subsector"
    boundaries_gdf = add_lon_lat_from_centroid(boundaries_gdf, index_col="loc_id")

    # noinspection PyTypeChecker
    gdf: gpd.GeoDataFrame = boundaries_gdf[LOCATIONS_COLUMNS].copy()

    return check_schema(gdf, LOCATIONS_SCHEMA)


# =========================================
# Journeys
# =========================================


def build_toronto_journeys(inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Parse raw trip records into standardised journey DF conforming to ``USER_JOURNEY_SCHEMA``."""
    trips = inputs.raw_journeys_df
    persons = inputs.raw_person_df

    # Replace app_id with user_id
    app_to_user_id_map = persons.select("app_id", "person_id").unique()
    user_journeys_df = trips.join(app_to_user_id_map, on="app_id")

    # Replace transport modes
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),
        pl.col("manual_mode").replace_strict(MANUAL_MODE_MAP, default=None),
    ).with_columns(
        pl
        .when(pl.col("manual_mode").is_not_null())
        .then(pl.col("manual_mode"))
        .otherwise("predicted_modes")
        .alias("leg_mode")
    )

    # Rank trip legs by start time
    leg_rank_expr = pl.col("start_fmt_time_section").rank("dense").over("app_id", "trip_id") - 1

    # Replace trip purposes
    purpose_expr = (
        pl.col("manual_purpose").fill_null(Purpose.UNKNOWN).replace_strict(PURPOSE_MAP, default=Purpose.OTHER)
    )

    user_journeys_df = user_journeys_df.select(
        user_id="person_id",
        journey_id="trip_id",
        leg_id=leg_rank_expr.cast(pl.Int8),
        leg_mode=pl.col("leg_mode").cast(pl.Categorical),
        leg_line=pl.lit(None, dtype=pl.String),
        dep_datetime=pl.col("start_fmt_time_section").str.to_datetime(format="%+", time_zone="EST"),
        dep_loc_id=pl.col("start_loc_CT_section"),
        arr_loc_id=pl.col("end_loc_CT_section"),
        arr_datetime=pl.col("end_fmt_time_section").str.to_datetime(format="%+", time_zone="EST"),
        arr_purpose=purpose_expr,
    )

    # Handle duration and start times/dates
    user_journeys_df = user_journeys_df.with_columns(
        dep_day=pl.col("dep_datetime").dt.date(),
        dep_time=pl.col("dep_datetime").dt.time(),
        duration=pl.col("arr_datetime") - pl.col("dep_datetime"),
    ).drop("dep_datetime", "arr_datetime")

    # Add departure purposes (purpose of previous trip)
    # Not a valid assumption, as trips are not necessarily linked => comment out
    """unique_trips = (
        user_journeys_df
        .sort("user_id", "dep_day", "dep_time")
        .unique(["user_id", "journey_id"], keep="first")
        .select(
            "journey_id",
            "arr_purpose",
            previous_journey_id=pl.col("journey_id").shift(1).over(["user_id", "dep_day"], order_by="dep_time"),
        )
    )

    dep_purposes = unique_trips.join(
        unique_trips.select("journey_id", dep_purpose="arr_purpose"),
        left_on="previous_journey_id",
        right_on="journey_id",
    ).drop("arr_purpose", "previous_journey_id")

    user_journeys_df = user_journeys_df.join(dep_purposes, on="journey_id", how="left").with_columns(
        pl.col("arr_purpose").cast(pl.Categorical),
        pl.col("dep_purpose").fill_null(Purpose.UNKNOWN).cast(pl.Categorical),
    )"""

    user_journeys_df = user_journeys_df.with_columns(
        pl.col("arr_purpose").cast(pl.Categorical),
        pl.lit(Purpose.UNKNOWN).alias("dep_purpose").cast(pl.Categorical),
    )

    # Replace null loc_ids with NA
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("dep_loc_id").fill_null(NA), pl.col("arr_loc_id").fill_null(NA)
    )

    # Replace unknown locations with NA
    locations_ids = locations_gdf["loc_id"]
    user_journeys_df = user_journeys_df.with_columns(
        dep_loc_id=pl.when(pl.col("dep_loc_id").is_in(locations_ids)).then(pl.col("dep_loc_id")).otherwise(pl.lit(NA)),
        arr_loc_id=pl.when(pl.col("arr_loc_id").is_in(locations_ids)).then(pl.col("dep_loc_id")).otherwise(pl.lit(NA)),
    )

    return user_journeys_df.select(USER_JOURNEY_SCHEMA.keys()).sort(
        "user_id",
        "dep_day",
        "dep_time",
    )


# =========================================
# Activities
# =========================================


def build_toronto_activities(inputs: TorontoInputs, user_journeys_df: pl.DataFrame):
    """Parse the wide-format hourly activity DF into a long, collapsed by span, activity DataFrame."""
    # Move from a wide to long data format by adding an "index" column for activities that happen in the same hour
    activities = unpivot_activities(inputs.raw_activities_df)

    # Only keep users with recorded trips
    user_ids = user_journeys_df["user_id"].unique()
    activities = activities.filter(pl.col("person_id").is_in(user_ids.implode()))

    # Parse Hour column into a day and hour column
    activities = (
        activities
        .with_columns(
            act_hour=pl.col("Hour").str.split("-").list.first(),
            act_day=pl.col("Hour").str.split("d").list.last().cast(int),
        )
        .with_columns(
            (pl.col("act_hour").str.pad_start(2, "0") + ":00").str.to_time("%H:%M"), pl.col("act_day").cast(int)
        )
        .drop("Hour")
    )

    # Replace day numbering with actual activity dates
    start_dates = inputs.raw_person_df.select(
        "person_id", start_date=pl.col("THATS Experiment start date").str.to_date("%Y-%m-%d")
    ).lazy()

    activities = (
        activities
        .join(start_dates, on="person_id", how="left")
        .with_columns(act_day=pl.col("start_date") + pl.duration(days="act_day"))
        .drop("start_date")
    )

    row_index_cols = ["hh_id", "person_id", "act_day", "act_hour", "act_index"]
    value_cols = [col for col in activities.collect_schema().names() if col not in row_index_cols]

    activities = activities.sort(row_index_cols).select(*row_index_cols, *value_cols).rename({"act_day": "act_date"})

    # Replace activity type with Purposes
    activities = activities.with_columns(
        pl.col("ActivityType").cast(float).cast(int).replace_strict(ACTIVITY_MAP, return_dtype=pl.Categorical)
    ).rename({"ActivityType": "act_purpose"})

    # Collapse hourly breakdown into list of activities with start and end time
    activities = collapse_activities(activities)

    # Filter out empty purpose activities
    activities = activities.drop_nulls("act_purpose")

    return activities.collect()


def unpivot_activities(activity_df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert the wide per-hour, per-index activity columns into a long LazyFrame."""
    value_cols = [
        "StrtTripID",
        "EndTripID",
        "ActivityType",
        "ActivityTypeCategory",
        "ActivityWithPartner",
        "ActivityWithChild",
        "ActivityWithOtherRel",
        "ActivityWithFriends",
        "ActivityWithOther",
        "ActivityHorizon",
        "ActivityExpenditure",
    ]

    row_indentifiers = ["hh_id", "person_id", "Hour"]
    primary_key = [*row_indentifiers, "act_index"]
    unpivoted_df = (
        activity_df
        .select(*row_indentifiers, act_index=list(range(1, 8)))
        .explode("act_index")
        .unique(maintain_order=True)
    )

    for value_name in value_cols:
        df = (
            activity_df
            .select(*row_indentifiers, cs.ends_with(value_name))
            .unpivot(
                index=row_indentifiers,
                variable_name="act_index",
                value_name=value_name,
            )
            .with_columns(pl.col("act_index").str.slice(0, 1).cast(int))
        )

        unpivoted_df = unpivoted_df.join(df, on=primary_key, how="left")

    return unpivoted_df


def collapse_activities(activity_df: pl.LazyFrame) -> pl.LazyFrame:
    """Merge consecutive same-purpose hourly activity rows into contiguous spans with start/end times."""
    # Find the hour of the previous activity with the same purpose that day
    activities = activity_df.with_columns(
        prev_hour_same_purpose=pl.col("act_hour").shift(1).over("hh_id", "person_id", "act_date", "act_purpose")
    )

    # Compute the hour gap between the current activity and the previous activity with the same purpose
    activities = activities.with_columns(
        hour_gap=pl
        .when(pl.col("prev_hour_same_purpose").is_not_null())
        .then(
            (
                pl.col("act_hour").cast(pl.Duration("ms")) - pl.col("prev_hour_same_purpose").cast(pl.Duration("ms"))
            ).dt.total_hours()
        )
        .otherwise(None)
    )

    # Indicate if there is a change in activity (hour gap larger than 1) and identify the spans
    activities = activities.with_columns(
        is_new_span=pl.col("hour_gap").is_null() | (pl.col("hour_gap") > 1)
    ).with_columns(span_id=pl.col("is_new_span").cum_sum().over("hh_id", "person_id", "act_date", "act_purpose"))

    # Group by span and find the start, end times and duration
    activities = (
        activities
        .group_by("hh_id", "person_id", "act_date", "act_purpose", "span_id")
        .agg(
            start_time=pl.col("act_hour").min(),
            end_time=pl.col("act_hour").max(),
            n_hours=pl.col("act_hour").n_unique(),
        )
        .drop("span_id")
    )

    return activities.sort("hh_id", "person_id", "act_date", "start_time")


# =========================================
# Users
# =========================================


def build_toronto_users(
    inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame, user_journeys_df: pl.DataFrame
) -> pl.DataFrame:
    """Build the user DataFrame with household demographics and home location for Toronto."""
    persons = inputs.raw_person_df.select(
        "person_id", "hh_id", has_driving_license="THATS driverslicence", has_pt_pass="THATS transitpass"
    )  # TODO add demographics: HH role, age, gender, education, employment status, student status, driving license, PT pass.

    hhs = inputs.raw_household_df.select(
        "hh_id",
        home_loc_id="THATS HomeCT",
        hh_num_adults="THATS NumAdults",
        hh_num_children="THATS NumChildren",
        hh_num_vehicles="THATS NumVeh",
        hh_num_bikes="THATS NumBike",
    )  # TODO add HH demographics: HH size (num adults, num children), HH income, HH location, num vehicles, num bikes.

    demographics = persons.join(hhs, on="hh_id", how="left").drop("hh_id")
    user_ids = user_journeys_df.select("user_id").unique()

    # Replace unknown locations with NA
    locations_ids = locations_gdf["loc_id"]
    demographics = demographics.with_columns(
        home_loc_id=pl.when(pl.col("home_loc_id").is_in(locations_ids)).then("home_loc_id").otherwise(pl.lit(NA))
    ).with_columns(pl.col("home_loc_id").fill_null(NA))

    return user_ids.join(demographics, left_on="user_id", right_on="person_id")
