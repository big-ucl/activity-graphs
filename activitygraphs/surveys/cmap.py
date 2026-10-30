"""Chicago CMAP MyDailyTravel survey: raw tables -> ``CMAPSurvey``, and the census statistics for the graph nodes.

Survey: ``CMAPSurvey.load`` reads the processed cache, or else calls ``read_tables`` and ``build_cmap_survey``,
which builds the locations, journeys and users in turn.
Statistics: ``read_cmap_statistics`` and ``add_cmap_population_job_statistics`` add census population and jobs to
the zones. ``LOADER`` bundles these steps for the survey registry.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import geopandas as gpd
import pandas as pd
import polars as pl

from activitygraphs.base import (
    CRS,
    LOCATIONS_COLUMNS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    USER_SCHEMA,
    Mode,
    Purpose,
)
from activitygraphs.config import CMAPDataConfig
from activitygraphs.schema import TensorSchema
from activitygraphs.surveys.base import NA, DataFrameStore, SurveyData, SurveyLoader, build_special_locations
from activitygraphs.surveys.demographics import DEMOGRAPHICS_SCHEMA, INCOME_QUARTILES, quantile_bands
from activitygraphs.surveys.statistics import normalise_statistics
from activitygraphs.utils import (
    add_lon_lat_from_centroid,
    check_schema,
    get_project_root,
    invert_mapping,
)

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

INCOME_BANDS_KUSD = {
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
}

AGE_BAND_MAP = invert_mapping({
    "0-17": [1, 2, 3, 4],
    "18-24": [5],
    "25-34": [6],
    "35-44": [7],
    "45-54": [8],
    "55-64": [9],
    "65-74": [10],
    "75+": [11, 12],
})

AGE_CODES_UNDER_5 = [1]
AGE_CODES_UNDER_12 = [1, 2]
AGE_CODES_UNDER_16 = [1, 2, 3]

EMPLOYMENT_MAP = invert_mapping({
    "full_time": [1],
    "part_time": [2],
    "self_employed": [3],
    "not_employed": [5, 6, 7, 8, 9, 10, 11],
})

STUDENT_MAP = invert_mapping({
    "full_time": [0, 4],
    "part_time": [1, 3],
    "not_student": [2],
})

TRANSIT_PASS_COLUMNS = [f"transit_pass_{option}" for option in range(1, 6)]


# =========================================
# Survey
# =========================================


@dataclass(frozen=True)
class CMAPTables:
    """Parsed raw inputs for the CMAP MyDailyTravel survey (journeys, persons, households, days, boundaries)."""

    raw_journeys_df: pl.DataFrame
    raw_persons_df: pl.DataFrame
    raw_household_df: pl.DataFrame

    boundaries_ct: gpd.GeoDataFrame


class CMAPSurvey(SurveyData, DataFrameStore):
    """Chicago CMAP MyDailyTravel ``SurveyData`` subclass with caching via ``DataFrameStore``."""

    def __init__(
        self,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        users_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        super().__init__(user_journeys_df, locations_gdf, filters)
        self._users_df = users_df

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        """Home location of each user, one row per user."""
        return self.users_df.select("user_id", loc_id="home_loc_id")

    @cached_property
    def users_df(self) -> pl.DataFrame:
        """Users, filtered by active home location filters, sorted by ``user_id``."""
        return self._filter_loc_types(self._users_df, "home_loc_id").sort("user_id")

    @cached_property
    def user_ids(self) -> pl.Series:
        """Sorted ids of the users in ``users_df``."""
        return self.users_df["user_id"].sort()

    @property
    def demographics_schema(self) -> TensorSchema:
        """Schema of the demographic columns in ``users_df``, in tensor column order."""
        return DEMOGRAPHICS_SCHEMA

    def _copy(self, filters: list[str] | None = None) -> Self:
        return type(self)(self._locations_gdf, self._user_journeys_df, self.users_df, filters)

    @classmethod
    def load(cls, cfg: CMAPDataConfig, project_root: Path | None = None, name: str | None = None) -> "CMAPSurvey":
        """Load the survey from the processed cache, building and caching it on first call."""
        project_root, data_dir = cls._dirs(cfg, project_root, name)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)
            users_df = pl.read_parquet(data_dir / "users_df.parquet")

            return cls(locations_gdf, user_journeys_df, users_df)

        survey = build_cmap_survey(read_tables(cfg, project_root))
        survey.save(cfg, project_root, name)

        return survey

    def save(self, cfg: CMAPDataConfig, project_root: Path | None = None, name: str | None = None):
        """Write the locations, journeys and users to the processed cache."""
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self._locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self._user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")
        self._users_df.write_parquet(data_dir / "users_df.parquet")


def read_tables(cfg: CMAPDataConfig, project_root: Path | None = None) -> CMAPTables:
    """Read the raw CMAP trip, person and household files and the census-tract boundaries."""
    project_root = get_project_root(project_root)

    raw_data_dir = project_root / cfg.paths.raw
    raw_journeys_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_journeys, infer_schema_length=None)
    raw_persons_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_person, infer_schema_length=10000)
    raw_household_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_household, infer_schema_length=10000)

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    boundaries_ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)

    return CMAPTables(raw_journeys_df, raw_persons_df, raw_household_df, boundaries_ct)


def build_cmap_survey(inputs: CMAPTables) -> CMAPSurvey:
    """Parse raw CMAP inputs into a standardised ``CMAPSurvey`` instance."""
    locations_gdf = build_cmap_locations(inputs)
    user_journeys_df = build_cmap_journeys(inputs, locations_gdf)
    users_df = build_cmap_users(inputs, locations_gdf, user_journeys_df)

    # Filter out journeys from excluded users
    user_journeys_df = user_journeys_df.join(users_df.select("user_id"), on="user_id", how="inner")

    return CMAPSurvey(locations_gdf, user_journeys_df, users_df)


# =========================================
# Statistics
# =========================================


@dataclass(frozen=True)
class CMAPStatistics:
    """ACS tract population and LODES workplace job tables for the CMAP region."""

    population_df: pl.DataFrame
    jobs_df: pl.DataFrame


def read_cmap_statistics(cfg: CMAPDataConfig, project_root: Path | None = None) -> CMAPStatistics:
    """Read the ACS population and LODES workplace-area CSVs."""
    stats_cfg = cfg.inputs.statistics
    stats_dir = get_project_root(project_root) / stats_cfg.directory

    population_df = pl.read_csv(stats_dir / stats_cfg.population, skip_rows_after_header=1)
    jobs_df = pl.read_csv(stats_dir / stats_cfg.jobs, schema_overrides={"w_geocode": pl.String})

    return CMAPStatistics(population_df, jobs_df)


def add_cmap_population_job_statistics(
    locations_gdf: gpd.GeoDataFrame, statistics: CMAPStatistics, normalise: bool = True
) -> gpd.GeoDataFrame:
    """Add census population and job counts to Chicago CMAP MyDailyTravel locations (census tracts).

    Args:
        locations_gdf: CMAP locations (must have a ``loc_id`` matching ``GEO_ID``).
        statistics: Census tables from ``read_cmap_statistics``.
        normalise: If True, divide counts by area to obtain densities.
    """
    utm_crs = locations_gdf.estimate_utm_crs()

    pop_stats = statistics.population_df.select(
        loc_id=pl.col("GEO_ID").str.split("US").list.last(), population=pl.col("B01003_001E")
    )

    job_stats = statistics.jobs_df.with_columns(loc_id=pl.col("w_geocode").str.slice(0, 11))
    job_stats = job_stats.group_by("loc_id").agg(jobs=pl.col("C000").sum())

    locations_gdf = locations_gdf.merge(pop_stats.to_pandas(), on="loc_id", how="left")
    locations_gdf = locations_gdf.merge(job_stats.to_pandas(), on="loc_id", how="left")

    locations_gdf[["population", "jobs"]] = locations_gdf[["population", "jobs"]].fillna(0)
    locations_gdf["area"] = locations_gdf.to_crs(utm_crs).geometry.area

    return normalise_statistics(locations_gdf, normalise)


# =========================================
# Loader
# =========================================


LOADER = SurveyLoader(
    load=CMAPSurvey.load,
    read_statistics=read_cmap_statistics,
    add_statistics=add_cmap_population_job_statistics,
)


# =========================================
# Survey: Locations
# =========================================


def build_cmap_locations(inputs: CMAPTables) -> gpd.GeoDataFrame:
    """Build the CMAP locations GeoDataFrame (census tracts and NA)."""
    zone_locations = inputs.boundaries_ct[inputs.boundaries_ct["COUNTYFP"].isin(CMAP_COUNTY_FP_CODES)]
    zone_locations = zone_locations.rename(columns={"GEOID": "loc_id", "NAME": "loc_name"}).to_crs(CRS)
    zone_locations["type"] = "zone"
    zone_locations = add_lon_lat_from_centroid(zone_locations, index_col="loc_id")
    zone_locations = zone_locations[LOCATIONS_COLUMNS]
    zone_locations = zone_locations[~zone_locations["loc_id"].isin(EXCLUDED_CENSUS_TRACTS)]

    special_locations = build_special_locations()

    locations_gdf = gpd.GeoDataFrame(
        pd.concat([special_locations, zone_locations], ignore_index=True),
        crs=CRS,
    )

    return check_schema(locations_gdf, LOCATIONS_SCHEMA)


def _census_tract_to_loc_id(census_tract_col: str, locations_gdf: gpd.GeoDataFrame) -> pl.Expr:
    loc_ids = set(locations_gdf["loc_id"])

    tract = pl.col(census_tract_col).cast(pl.String)
    return pl.when(tract.is_in(loc_ids)).then(tract).otherwise(pl.lit(NA))


# =========================================
# Survey: Journeys
# =========================================


def build_cmap_journeys(inputs: CMAPTables, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Build the standardised journey table from the raw linked trips, one leg per trip."""
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
# Survey: Users
# =========================================


def build_cmap_users(
    inputs: CMAPTables, locations_gdf: gpd.GeoDataFrame, user_journeys_df: pl.DataFrame
) -> pl.DataFrame:
    """Build one row per user with home tract, household id and demographics, single-day diaries only.

    The demographics follow the categories of ``DEMOGRAPHICS_SCHEMA``. Children with missing statuses follow the TTS
    assumptions:
     - Under-16s are not employed and cannot drive.
     - Under-5s are not students,
     - and persons aged 5 to 15 are full-time students.
     - Under-12s with no transit pass answer have no pass.
    The income quartiles are taken over the single-day households.
    """
    raw_hh_df = inputs.raw_household_df.filter(pl.col("diary_platform").is_in(SINGLE_DAY_DIARY_PLATFORMS))
    income_quartiles = quantile_bands(raw_hh_df["income_detailed"], list(INCOME_BANDS_KUSD), INCOME_QUARTILES)
    hh_df = raw_hh_df.select(
        hh_id=pl.col("hh_id").cast(pl.String),
        home_loc_id=_census_tract_to_loc_id("home_tract_2020", locations_gdf),
        hh_num_adults=pl.col("num_adults").cast(pl.UInt32),
        hh_num_children=pl.col("num_kids").cast(pl.UInt32),
        hh_num_vehicles=pl.col("num_vehicles").cast(pl.UInt32),
        income_quartile=pl.col("income_detailed").replace_strict(
            income_quartiles, default=None, return_dtype=pl.String
        ),
    )

    age = pl.col("age")
    persons_df = inputs.raw_persons_df.select(
        user_id=pl.col("person_id").cast(pl.String),
        hh_id=pl.col("hh_id").cast(pl.String),
        age_band=age.replace_strict(AGE_BAND_MAP, default=None, return_dtype=pl.String),
        employment_status=pl
        .when(age.is_in(AGE_CODES_UNDER_16))
        .then(pl.lit("not_employed"))
        .otherwise(pl.col("employment").replace_strict(EMPLOYMENT_MAP, default=None, return_dtype=pl.String)),
        student_status=pl
        .when(age.is_in(AGE_CODES_UNDER_5))
        .then(pl.lit("not_student"))
        .when(age.is_in(AGE_CODES_UNDER_16))
        .then(pl.lit("full_time"))
        .otherwise(pl.col("student").replace_strict(STUDENT_MAP, default=None, return_dtype=pl.String)),
        can_drive=pl
        .when(age.is_in(AGE_CODES_UNDER_16))
        .then(pl.lit(False))
        .when(pl.col("can_drive").is_in([0, 1]))
        .then(pl.col("can_drive") == 1)
        .otherwise(pl.lit(None, dtype=pl.Boolean)),
        has_transit_pass=pl
        .when(pl.any_horizontal(pl.col(TRANSIT_PASS_COLUMNS) == 1))
        .then(pl.lit(True))
        .when(pl.col("transit_pass_0") == 1)
        .then(pl.lit(False))
        .when(age.is_in(AGE_CODES_UNDER_12))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean)),
    )

    # Filter out users which have only journeys to NA
    journey_user_ids = user_journeys_df.filter((pl.col("dep_loc_id") != NA) | (pl.col("arr_loc_id") != NA))
    journey_user_ids = journey_user_ids.select("user_id").unique()

    users_df = persons_df.join(hh_df, on="hh_id", how="inner")
    users_df = users_df.join(journey_user_ids, on="user_id", how="inner")  # Drop users with no journeys in study area
    users_df = users_df.sort("user_id")

    return check_schema(users_df, USER_SCHEMA)
