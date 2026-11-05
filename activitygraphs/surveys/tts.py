"""Transportation Tomorrow Survey (TTS), 2016 and 2022 survey versions: raw tables -> ``TTSSurvey``.

``TTSSurvey.load`` reads the processed cache, or else calls ``read_tables`` and ``_build_tts_survey``, which builds
the locations, journeys and users in turn. The zones are the 2006 traffic zones of the six GTHA regions. Both survey
versions are coded to these zones, and the 2022 purposes are collapsed onto the 2016 purpose codes.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import city2graph as c2g
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
from activitygraphs.config import TTSDataConfig
from activitygraphs.schema import TensorSchema
from activitygraphs.surveys.base import NA, DataFrameStore, SurveyData, SurveyLoader, build_special_locations
from activitygraphs.surveys.demographics import DEMOGRAPHICS_SCHEMA, INCOME_QUARTILES, quantile_bands, vehicle_band
from activitygraphs.surveys.statistics import normalise_statistics, overlap_counts
from activitygraphs.utils import add_lon_lat_from_centroid, check_schema, get_project_root, invert_mapping

GTHA_REGIONS = {
    "Toronto": 1,
    "Durham": 2,
    "York": 3,
    "Peel": 4,
    "Halton": 5,
    "Hamilton": 6,
}

ZONE_ID_COLUMN = "gta06"
REGION_COLUMN = "region"

ISLAND_NEIGHBOURS = 3

DA_ID_COLUMN = "DAUID"
PROVINCE_COLUMN = "PRUID"
ONTARIO_PROVINCE_ID = "35"

MODE_MAP = invert_mapping({
    Mode.BUS: ["B", "S"],
    Mode.TRAIN: ["G", "J"],
    Mode.CYCLE: ["C"],
    Mode.CAR: ["D"],
    Mode.VEH_PASS: ["P"],
    Mode.MOTORCYCLE: ["M"],
    Mode.TAXI: ["T", "U"],
    Mode.WALK: ["W"],
    Mode.OTHER: ["O", "E"],
    Mode.UNKNOWN: ["9"],
})

PURPOSE_MAP_2016 = invert_mapping({
    Purpose.HOME: ["H"],
    Purpose.WORK_MAIN: ["W", "R"],
    Purpose.STUDY: ["S", "C"],
    Purpose.ESCORT: ["D", "F"],
    Purpose.SHOP: ["M"],
    Purpose.OTHER: ["O"],
    Purpose.UNKNOWN: ["9"],
})

PURPOSE_MAP_2022 = invert_mapping({
    Purpose.HOME: [80],
    Purpose.WORK_MAIN: [10, 11, 12],
    Purpose.STUDY: [20],
    Purpose.ESCORT: [30, 51, 52],
    Purpose.SHOP: [40, 41, 42, 44],
    Purpose.OTHER: [43, 45, 46, 47, 77, 777],
    Purpose.UNKNOWN: [999],
})

AGE_BAND_MAP = {
    "0-12": "0-17",
    "13-17": "0-17",
    "18-24": "18-24",
    "25-34": "25-34",
    "35-44": "35-44",
    "45-55": "45-54",
    "55-64": "55-64",
    "65-74": "65-74",
    "75-99": "75+",
}

CHILD_AGE_RANGES = ["0-12", "13-17"]

EMPLOYMENT_MAP = invert_mapping({
    "full_time": ["F", "H"],
    "part_time": ["P", "J"],
    "not_employed": ["O"],
})

STUDENT_MAP = {"S": "full_time", "P": "part_time", "O": "not_student"}

DRIVER_LICENCE_MAP = {"Y": True, "N": False, "0": False}

TRANSIT_PASS_MAP = {"C": True, "G": True, "M": True, "O": True, "P": True, "Y": True, "N": False, "X": False}

INCOME_BAND_CODES = [1, 2, 3, 4, 5, 6]

UNKNOWN_VEHICLE_COUNT = 99
UNKNOWN_START_TIME = 9999

PERSON_CODE_COLUMNS = ["age_range", "driver_lic", "tran_pass", "emp_stat", "stu_stat"]


@dataclass(frozen=True)
class CensusPopulationFormat:
    """The columns of a census profile CSV that hold the population of each dissemination area.

    Attributes:
        da_id_col: Column of the dissemination area id.
        characteristic_col: Column of the characteristic id of each row.
        population_characteristic_id: Characteristic id of the population rows.
        value_col: Column of the count.
    """

    da_id_col: str
    characteristic_col: str
    population_characteristic_id: int
    value_col: str


CENSUS_2016_POPULATION = CensusPopulationFormat(
    "GEO_CODE (POR)",
    "Member ID: Profile of Dissemination Areas (2247)",
    1,
    "Dim: Sex (3): Member ID: [1]: Total - Sex",
)
CENSUS_2021_POPULATION = CensusPopulationFormat("ALT_GEO_CODE", "CHARACTERISTIC_ID", 1, "C1_COUNT_TOTAL")


@dataclass(frozen=True)
class TTSVersionFormat:
    """The data format of a TTS survey version: i.e. trip purpose columns and census population format.

    Attributes:
        purpose_orig_col: Column of the origin purpose of each trip.
        purpose_dest_col: Column of the destination purpose of each trip.
        purpose_dtype: Dtype of the purpose codes.
        purpose_map: Map from the purpose codes to ``Purpose``.
        census_population: Format of the census profile CSV read for the zone population.
    """

    purpose_orig_col: str
    purpose_dest_col: str
    purpose_dtype: type[pl.DataType]
    purpose_map: dict
    census_population: CensusPopulationFormat


VERSION_FORMATS = {
    2016: TTSVersionFormat("purp_orig", "purp_dest", pl.String, PURPOSE_MAP_2016, CENSUS_2016_POPULATION),
    2022: TTSVersionFormat("purp_orig2022", "purp_dest2022", pl.Int64, PURPOSE_MAP_2022, CENSUS_2021_POPULATION),
}


# =========================================
# Survey
# =========================================


@dataclass(frozen=True)
class TTSTables:
    """Parsed raw inputs of one TTS survey version (trips, persons, households, zone boundaries)."""

    raw_journeys_df: pl.DataFrame
    raw_persons_df: pl.DataFrame
    raw_household_df: pl.DataFrame

    boundaries_zones: gpd.GeoDataFrame
    survey_version: int


class TTSSurvey(SurveyData, DataFrameStore):
    """TTS ``SurveyData`` subclass with caching via ``DataFrameStore``."""

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
    def load(cls, cfg: TTSDataConfig, project_root: Path | None = None, name: str | None = None) -> "TTSSurvey":
        """Load the survey from the processed cache, building and caching it on first call."""
        project_root, data_dir = cls._dirs(cfg, project_root, name)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)
            users_df = pl.read_parquet(data_dir / "users_df.parquet")

            return cls(locations_gdf, user_journeys_df, users_df)

        survey = _build_tts_survey(read_tables(cfg, project_root))
        survey.save(cfg, project_root, name)

        return survey

    def save(self, cfg: TTSDataConfig, project_root: Path | None = None, name: str | None = None):
        """Write the locations, journeys and users to the processed cache."""
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self._locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self._user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")
        self._users_df.write_parquet(data_dir / "users_df.parquet")


def read_tables(cfg: TTSDataConfig, project_root: Path | None = None) -> TTSTables:
    """Read the trip, person and household tables of the survey version of ``cfg``, and the traffic zone boundaries."""
    project_root = get_project_root(project_root)
    version_format = VERSION_FORMATS[cfg.survey_version]

    raw_data_dir = project_root / cfg.paths.raw
    trip_overrides = {
        "mode_prime": pl.String,
        version_format.purpose_orig_col: version_format.purpose_dtype,
        version_format.purpose_dest_col: version_format.purpose_dtype,
    }
    person_overrides = dict.fromkeys(PERSON_CODE_COLUMNS, pl.String)

    raw_journeys_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_journeys, schema_overrides=trip_overrides)
    raw_persons_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_person, schema_overrides=person_overrides)
    raw_household_df = pl.read_csv(raw_data_dir / cfg.inputs.raw_household)

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    boundaries_zones = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.zones)

    return TTSTables(raw_journeys_df, raw_persons_df, raw_household_df, boundaries_zones, cfg.survey_version)


def _build_tts_survey(inputs: TTSTables) -> TTSSurvey:
    """Parse the raw inputs of one TTS survey version into a standardised ``TTSSurvey`` instance."""
    locations_gdf = _build_tts_locations(inputs)
    user_journeys_df = _build_tts_journeys(inputs, locations_gdf)
    users_df = _build_tts_users(inputs, locations_gdf, user_journeys_df)

    user_journeys_df = user_journeys_df.join(users_df.select("user_id"), on="user_id", how="inner")

    return TTSSurvey(locations_gdf, user_journeys_df, users_df)


# =========================================
# Statistics
# =========================================


@dataclass(frozen=True)
class TTSStatistics:
    """The census population and the polygon of each Ontario dissemination area."""

    dissemination_areas: gpd.GeoDataFrame


def _read_tts_statistics(cfg: TTSDataConfig, project_root: Path | None = None) -> TTSStatistics:
    """Read the population of the Ontario dissemination areas, from the census of the survey version of ``cfg``.

    Raises:
        ValueError: No dissemination area of the boundary file has a population row.
    """
    project_root = get_project_root(project_root)
    census = VERSION_FORMATS[cfg.survey_version].census_population

    boundaries_path = project_root / cfg.inputs.boundaries.directory / cfg.inputs.boundaries.dissemination_areas
    areas = gpd.read_file(boundaries_path)
    areas = areas[areas[PROVINCE_COLUMN] == ONTARIO_PROVINCE_ID][[DA_ID_COLUMN, "geometry"]]

    population_path = project_root / cfg.inputs.statistics.directory / cfg.inputs.statistics.population
    population = _read_census_population(population_path, census).to_pandas()
    areas = areas.merge(population, on=DA_ID_COLUMN, how="inner")

    if areas.empty:
        raise ValueError(f"No dissemination area of `{boundaries_path.name}` has a row in `{population_path.name}`.")

    return TTSStatistics(areas)


def _add_tts_population_statistics(
    locations_gdf: gpd.GeoDataFrame, statistics: TTSStatistics, normalise: bool = True
) -> gpd.GeoDataFrame:
    """Add the census population of each TTS zone, splitting each dissemination area by its share of area in the zone.

    Args:
        locations_gdf: TTS zones, with a ``loc_id`` column, in a projected CRS.
        statistics: Census population from ``_read_tts_statistics``.
        normalise: If True, divide the population by the zone area to obtain a density.
    """
    utm_crs = locations_gdf.estimate_utm_crs()
    population = overlap_counts(locations_gdf, statistics.dissemination_areas, "population")

    locations_gdf = locations_gdf.copy()
    locations_gdf["population"] = population.to_numpy()
    locations_gdf["area"] = locations_gdf.to_crs(utm_crs).geometry.area

    return normalise_statistics(locations_gdf, normalise, columns=("population",))


def _read_census_population(path: Path, census: CensusPopulationFormat) -> pl.DataFrame:
    """Return the population of each dissemination area of a census profile CSV, 0 where the count is suppressed."""
    population = pl.col(census.value_col).cast(pl.Float64, strict=False).fill_null(0.0)

    return (
        pl
        .scan_csv(path, encoding="utf8-lossy", infer_schema=False)
        .filter(pl.col(census.characteristic_col) == str(census.population_characteristic_id))
        .select(pl.col(census.da_id_col).alias(DA_ID_COLUMN), population=population)
        .collect()
    )


# =========================================
# Island edges
# =========================================


def _build_tts_island_edges(nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build kNN edges joining each zone without a contiguous neighbour to its ``ISLAND_NEIGHBOURS`` nearest zones.

    Args:
        nodes_gdf: Nodes of the contiguity graph, indexed by ``loc_id``.
        edges_gdf: Edges of the contiguity graph, indexed by (source ``loc_id``, target ``loc_id``).

    Returns:
        Edges indexed by (source ``loc_id``, target ``loc_id``), none when every zone has a contiguous neighbour.
    """
    linked = edges_gdf.index.get_level_values(0).union(edges_gdf.index.get_level_values(1))
    is_island = ~nodes_gdf.index.isin(linked)
    if not is_island.any():
        return edges_gdf.iloc[0:0]

    _, island_edges = c2g.knn_graph(nodes_gdf[is_island], k=ISLAND_NEIGHBOURS, target_gdf=nodes_gdf[~is_island])
    island_edges = island_edges.reset_index()
    island_edges["source"] = island_edges["source"].str[1]
    island_edges["target"] = island_edges["target"].str[1]

    return island_edges.set_index(["source", "target"])


# =========================================
# Loader
# =========================================


LOADER = SurveyLoader(
    load=TTSSurvey.load,
    read_statistics=_read_tts_statistics,
    add_statistics=_add_tts_population_statistics,
    build_extra_edges=_build_tts_island_edges,
)


# =========================================
# Survey: Locations
# =========================================


def _build_tts_locations(inputs: TTSTables) -> gpd.GeoDataFrame:
    """Build the TTS locations GeoDataFrame (the traffic zones of the six GTHA regions, and NA).

    Invalid zone geometries are repaired, keeping only their polygonal parts.
    """
    zones = inputs.boundaries_zones
    in_gtha = zones[REGION_COLUMN].isin(GTHA_REGIONS.values())

    zone_locations = zones[in_gtha].copy()
    zone_locations.geometry = zone_locations.geometry.make_valid(method="structure", keep_collapsed=False)
    zone_locations = zone_locations.to_crs(CRS)
    zone_locations["loc_id"] = zone_locations[ZONE_ID_COLUMN].astype(int).astype(str)
    zone_locations["loc_name"] = zone_locations["loc_id"]
    zone_locations["type"] = "zone"
    zone_locations = add_lon_lat_from_centroid(zone_locations, index_col="loc_id")
    zone_locations = zone_locations[LOCATIONS_COLUMNS]

    special_locations = build_special_locations()

    locations_gdf = gpd.GeoDataFrame(
        pd.concat([special_locations, zone_locations], ignore_index=True),
        crs=CRS,
    )

    return check_schema(locations_gdf, LOCATIONS_SCHEMA)


def _zone_to_loc_id(zone_col: str, locations_gdf: gpd.GeoDataFrame) -> pl.Expr:
    loc_ids = set(locations_gdf["loc_id"])

    zone = pl.col(zone_col).cast(pl.String)
    return pl.when(zone.is_in(loc_ids)).then(zone).otherwise(pl.lit(NA))


# =========================================
# Survey: Journeys
# =========================================


def _build_tts_journeys(inputs: TTSTables, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Build the standardised journey table from the raw trips, one leg per trip."""
    version_format = VERSION_FORMATS[inputs.survey_version]
    start_time = pl.when(pl.col("start_time") != UNKNOWN_START_TIME).then(pl.col("start_time"))
    dep_time = pl.time((start_time // 100) % 24, start_time % 100)

    user_journeys_df = inputs.raw_journeys_df.sort("hp_num", "start_time").select(
        user_id=pl.col("hp_num").cast(pl.String),
        journey_id=pl.col("hpt_num").cast(pl.String),
        leg_id=pl.lit(1, dtype=pl.Int8),
        leg_mode=pl.col("mode_prime").replace_strict(MODE_MAP, default=Mode.UNKNOWN).cast(pl.Categorical),
        leg_line=pl.lit(None, dtype=pl.String),
        duration=pl.lit(None, dtype=pl.Duration()),
        dep_day=pl.lit(None, dtype=pl.Date),
        dep_time=dep_time,
        dep_purpose=pl
        .col(version_format.purpose_orig_col)
        .replace_strict(version_format.purpose_map, default=Purpose.UNKNOWN)
        .cast(pl.Categorical),
        dep_loc_id=_zone_to_loc_id("gta06_orig", locations_gdf),
        arr_loc_id=_zone_to_loc_id("gta06_dest", locations_gdf),
        arr_purpose=pl
        .col(version_format.purpose_dest_col)
        .replace_strict(version_format.purpose_map, default=Purpose.UNKNOWN)
        .cast(pl.Categorical),
    )

    return check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)


# =========================================
# Survey: Users
# =========================================


def _build_tts_users(
    inputs: TTSTables, locations_gdf: gpd.GeoDataFrame, user_journeys_df: pl.DataFrame
) -> pl.DataFrame:
    """Build one row per user with home zone, household id and demographics."""
    raw_hh_df = inputs.raw_household_df.with_columns(home_loc_id=_zone_to_loc_id("gta06_hhld", locations_gdf))
    zone_hh_df = raw_hh_df.filter(pl.col("home_loc_id") != NA)
    income_quartiles = quantile_bands(zone_hh_df["income"], INCOME_BAND_CODES, INCOME_QUARTILES)

    is_child = pl.col("age_range").is_in(CHILD_AGE_RANGES)
    household_sizes = inputs.raw_persons_df.group_by("hhld_num").agg(
        hh_num_adults=(~is_child).sum().cast(pl.UInt32),
        hh_num_children=is_child.sum().cast(pl.UInt32),
    )

    n_vehicle = pl.col("n_vehicle")
    known_vehicles = pl.when(n_vehicle != UNKNOWN_VEHICLE_COUNT).then(n_vehicle)
    hh_df = raw_hh_df.join(household_sizes, on="hhld_num", how="left").select(
        hh_id=pl.col("hhld_num").cast(pl.String),
        home_loc_id="home_loc_id",
        hh_num_adults="hh_num_adults",
        hh_num_children="hh_num_children",
        hh_num_vehicles=vehicle_band(known_vehicles),
        income_quartile=pl.col("income").replace_strict(income_quartiles, default=None, return_dtype=pl.String),
    )

    persons_df = inputs.raw_persons_df.select(
        user_id=pl.col("hp_num").cast(pl.String),
        hh_id=pl.col("hhld_num").cast(pl.String),
        age_band=pl.col("age_range").replace_strict(AGE_BAND_MAP, default=None, return_dtype=pl.String),
        employment_status=pl.col("emp_stat").replace_strict(EMPLOYMENT_MAP, default=None, return_dtype=pl.String),
        student_status=pl.col("stu_stat").replace_strict(STUDENT_MAP, default=None, return_dtype=pl.String),
        can_drive=pl.col("driver_lic").replace_strict(DRIVER_LICENCE_MAP, default=None, return_dtype=pl.Boolean),
        has_transit_pass=pl.col("tran_pass").replace_strict(TRANSIT_PASS_MAP, default=None, return_dtype=pl.Boolean),
    )

    journey_user_ids = user_journeys_df.filter((pl.col("dep_loc_id") != NA) | (pl.col("arr_loc_id") != NA))
    journey_user_ids = journey_user_ids.select("user_id").unique()

    users_df = persons_df.join(hh_df, on="hh_id", how="inner")
    users_df = users_df.join(journey_user_ids, on="user_id", how="inner")
    users_df = users_df.sort("user_id")

    return check_schema(users_df, USER_SCHEMA)
