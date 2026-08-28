"""Chicago CMAP MyDailyTravel survey loader and NetworkData subclass."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl

from activitygraphs.base import (
    CRS,
    LOCATIONS_COLUMNS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    Mode,
    Purpose,
    USER_SCHEMA,
)
from activitygraphs.config import CMAPDataConfig
from activitygraphs.network import NetworkData, build_special_locations, NA
from activitygraphs.utils import (
    DataFrameStore,
    add_lon_lat_from_centroid,
    check_schema,
    get_project_root,
    invert_mapping,
    bands_to_midpoint_map,
)

DEMOGRAPHIC_COLUMNS = [
    "has_missing_demographics",
    "age",
    "is_employed",
    "is_student",
    "can_drive",
    "hh_num_adults",
    "hh_num_children",
    "hh_num_vehicles",
    "hh_num_bicycle",
    "hh_income_kusd",
]

CMAP_COUNTY_FP_CODES = [
    "031",  # Cook
    "043",  # DuPage
    "089",  # Kane
    "093",  # Kendall
    "097",  # Lake
    "111",  # McHenry
    "197",  # Will
]

# Exclude the two Census Tracts inside the lake
EXCLUDED_CENSUS_TRACTS = ["17097990000", "17031990000"]

MODE_MAP = invert_mapping({
    Mode.OTHER: [16, 17],
    Mode.UNKNOWN: [-1],
    Mode.BOAT: [2, 5],
    Mode.BUS: [1, 4, 7, 18],
    Mode.WALK: [15],
    Mode.CYCLE: [11, 12],
    Mode.MOTORCYCLE: [],
    Mode.TAXI: [13, 14],
    Mode.TRAIN: [3, 6],
    Mode.TRAMWAY: [],
    Mode.VEH_PASS: [8, 9],
    Mode.CAR: [10],
})

PURPOSE_MAP = invert_mapping({
    Purpose.OTHER: [11, 13, 14],
    Purpose.UNKNOWN: [-1, 995],
    Purpose.HOME: [1],
    Purpose.WORK_MAIN: [2],
    Purpose.WORK_OTHER: [3],
    Purpose.STUDY: [4, 5],
    Purpose.VISIT: [12],
    Purpose.ESCORT: [6],
    Purpose.PERSONAL: [10],
    Purpose.SHOP: [7],
    Purpose.ENTERTAINMENT: [8],
    Purpose.LEISURE_OTHER: [9],
    Purpose.LONG_DISTANCE_TRIP: [],
})

SINGLE_DAY_DIARY_PLATFORMS = ["browser", "call"]

INCOME_MIDPOINTS_KUSD = bands_to_midpoint_map({
    1: (0, 15),
    2: (15, 25),
    3: (25, 30),
    4: (30, 35),
    5: (35, 50),
    6: (50, 60),
    7: (60, 75),
    8: (75, 100),
    9: (100, 150),
    10: (150, 200),
    11: (200, 250),
    12: (250, 350),
})

AGE_MIDPOINTS = bands_to_midpoint_map({
    1: (0, 5),
    2: (5, 11),
    3: (12, 15),
    4: (16, 17),
    5: (18, 24),
    6: (25, 34),
    7: (35, 44),
    8: (45, 54),
    9: (55, 64),
    10: (65, 74),
    11: (75, 84),
    12: (85, 100),
})


@dataclass(frozen=True)
class CMAPInputs:
    """Parsed raw inputs for the CMAP MyDailyTravel survey (journeys, persons, households, days, boundaries)."""

    raw_journeys_df: pl.DataFrame
    raw_persons_df: pl.DataFrame
    raw_household_df: pl.DataFrame

    boundaries_ct: gpd.GeoDataFrame


class CMAPData(NetworkData, DataFrameStore):
    """Chicago CMAP MyDailyTravel ``NetworkData`` subclass with caching via ``DataFrameStore``."""

    def __init__(
        self,
        inputs: CMAPInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        users_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        super().__init__(user_journeys_df, locations_gdf, filters)
        self.inputs = inputs
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

    @property
    def demographic_columns(self) -> list[str]:
        return DEMOGRAPHIC_COLUMNS

    def _copy(self, filters: list[str] | None = None):
        return CMAPData(self.inputs, self._locations_gdf, self._user_journeys_df, self.users_df, filters)

    @classmethod
    def load(cls, cfg: CMAPDataConfig, project_root: Path | None = None, name: str | None = None) -> "CMAPData":
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        cmap_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)
            users_df = pl.read_parquet(data_dir / "users_df.parquet")

            return cls(cmap_inputs, locations_gdf, user_journeys_df, users_df)
        else:
            data = build_cmap_data(cmap_inputs)
            data.save(cfg, project_root, name)

            return data

    def save(self, cfg: CMAPDataConfig, project_root: Path | None = None, name: str | None = None):
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self._locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self._user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")
        self._users_df.write_parquet(data_dir / "users_df.parquet")


def load_files(cfg: CMAPDataConfig, project_root: Path | None = None) -> CMAPInputs:
    project_root = get_project_root(project_root)

    raw_data_dir = project_root / cfg.paths.raw
    raw_journeys_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_journeys, infer_schema_length=None)
    raw_persons_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_person, infer_schema_length=10000)
    raw_household_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_household, infer_schema_length=10000)

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    boundaries_ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)

    return CMAPInputs(raw_journeys_df, raw_persons_df, raw_household_df, boundaries_ct)


def build_cmap_data(inputs: CMAPInputs) -> CMAPData:
    """Parse raw CMAP inputs into a standardised ``CMAPData`` instance."""
    locations_gdf = build_cmap_locations(inputs)
    user_journeys_df = build_cmap_journeys(inputs, locations_gdf)
    users_df = build_cmap_users(inputs, locations_gdf, user_journeys_df)

    # Filter out journeys from excluded users
    user_journeys_df = user_journeys_df.join(users_df.select("user_id"), on="user_id", how="inner")

    return CMAPData(inputs, locations_gdf, user_journeys_df, users_df)


# =========================================
# Locations
# =========================================


def build_cmap_locations(inputs: CMAPInputs) -> gpd.GeoDataFrame:
    """Build the CMAP locations GeoDataFrame (census tracts and NA)."""
    subsector_locations = inputs.boundaries_ct[inputs.boundaries_ct["COUNTYFP"].isin(CMAP_COUNTY_FP_CODES)]
    subsector_locations = subsector_locations.rename(columns={"GEOID": "loc_id", "NAME": "loc_name"}).to_crs(CRS)
    subsector_locations["type"] = "subsector"
    subsector_locations = add_lon_lat_from_centroid(subsector_locations, index_col="loc_id")
    subsector_locations = subsector_locations[LOCATIONS_COLUMNS]
    subsector_locations = subsector_locations[~subsector_locations["loc_id"].isin(EXCLUDED_CENSUS_TRACTS)]

    special_locations = build_special_locations()

    locations_gdf = gpd.GeoDataFrame(
        pd.concat([special_locations, subsector_locations], ignore_index=True),
        crs=CRS,
    )

    return check_schema(locations_gdf, LOCATIONS_SCHEMA)


def _census_tract_to_loc_id(census_tract_col: str, locations_gdf: gpd.GeoDataFrame) -> pl.Expr:
    loc_ids = set(locations_gdf["loc_id"])

    census_tract_col = pl.col(census_tract_col).cast(pl.String)
    return pl.when(census_tract_col.is_in(loc_ids)).then(census_tract_col).otherwise(pl.lit(NA))


# =========================================
# Journeys
# =========================================


def build_cmap_journeys(inputs: CMAPInputs, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    user_journeys_df = inputs.raw_journeys_df.select(
        user_id=pl.col("person_id").cast(pl.String),
        journey_id=pl.col("linked_trip_id").cast(pl.String),
        leg_id=pl.lit(1, dtype=pl.Int8),
        leg_mode=pl.col("linked_trip_mode").replace_strict(MODE_MAP, default=Mode.UNKNOWN).cast(pl.Categorical),
        leg_line=pl.lit(None, dtype=pl.String),
        duration=pl.duration(minutes=pl.col("duration_minutes")),
        dep_day=pl.col("depart_date").str.to_date(),
        dep_time=pl.time(pl.col("depart_hour"), pl.col("depart_minute"), pl.col("depart_seconds").cast(pl.Int64)),
        dep_purpose=pl
        .col("o_purpose_category")
        .replace_strict(PURPOSE_MAP, default=Purpose.UNKNOWN)
        .cast(pl.Categorical),
        dep_loc_id=_census_tract_to_loc_id("o_tract_2020", locations_gdf),
        arr_loc_id=_census_tract_to_loc_id("d_tract_2020", locations_gdf),
        arr_purpose=pl
        .col("d_purpose_category")
        .replace_strict(PURPOSE_MAP, default=Purpose.UNKNOWN)
        .cast(pl.Categorical),
    ).sort("user_id", "dep_day", "dep_time")

    return check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)


# =========================================
# Users
# =========================================


def build_cmap_users(
    inputs: CMAPInputs, locations_gdf: gpd.GeoDataFrame, user_journeys_df: pl.DataFrame
) -> pl.DataFrame:
    hh_df = inputs.raw_household_df.filter(pl.col("diary_platform").is_in(SINGLE_DAY_DIARY_PLATFORMS)).select(
        hh_id=pl.col("hh_id").cast(pl.String),
        home_loc_id=_census_tract_to_loc_id("home_tract_2020", locations_gdf),
        hh_num_adults=pl.col("num_adults").cast(pl.UInt32),
        hh_num_children=pl.col("num_kids").cast(pl.UInt32),
        hh_num_vehicles=pl.col("num_vehicles").cast(pl.UInt32),
        hh_num_bicycle=pl.col("num_bicycle").replace(995, None),
        hh_income_kusd=pl.col("income_detailed").replace_strict(INCOME_MIDPOINTS_KUSD, default=None),
    )

    persons_df = inputs.raw_persons_df.select(
        user_id=pl.col("person_id").cast(pl.String),
        hh_id=pl.col("hh_id").cast(pl.String),
        has_missing_demographics=(
            (pl.col("employment") == 995) | (pl.col("student") == 995) | (pl.col("can_drive") == 995)
        ).cast(pl.Boolean),
        age=pl.col("age").replace_strict(AGE_MIDPOINTS, default=None),
        is_employed=pl.col("employment").is_in([1, 2, 3, 8]).cast(pl.Boolean),
        is_student=pl.col("student").is_in([0, 1, 3, 4]).cast(pl.Boolean),
        can_drive=(pl.col("can_drive") == 1).cast(pl.Boolean),
    )

    # Filter out users which have only journeys to NA
    journey_user_ids = user_journeys_df.filter((pl.col("dep_loc_id") != NA) | (pl.col("arr_loc_id") != NA))
    journey_user_ids = journey_user_ids.select("user_id").unique()

    users_df = persons_df.join(hh_df, on="hh_id", how="inner")
    users_df = users_df.join(journey_user_ids, on="user_id", how="inner")  # Drop users with no journeys in study area

    users_df = users_df.with_columns(
        pl.col("age").fill_null(pl.col("age").median()).cast(pl.UInt32),
        pl.col("hh_income_kusd").fill_null(pl.col("hh_income_kusd").median()),
        pl.col("hh_num_bicycle").fill_null(pl.col("hh_num_bicycle").median()).cast(pl.UInt32),
    ).sort("user_id")

    return check_schema(users_df, USER_SCHEMA)
