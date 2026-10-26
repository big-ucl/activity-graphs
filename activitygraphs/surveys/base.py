"""SurveyData class that wraps the tables of different travel surveys in one interface, and SurveyLoader."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import geopandas as gpd
import polars as pl

from activitygraphs import utils
from activitygraphs.base import (
    CRS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    Purpose,
)
from activitygraphs.config import DataConfig
from activitygraphs.utils import check_schema

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"


class SurveyData(ABC):
    """Wrapper around survey DataFrames that provides a cached, filterable view.

    Each dataset (TPG, THATS, CMAP) inherits from this class.

    Locations are filtered by ``filters``. It will only show locations that match the `loc_type` provided in the filter.
    Only journeys with both endpoints in filtered locations and users with home locations inside filtered locations will not be shown.

    Use ``with_filter(loc_types)`` to create a restricted copy without mutating the original.
    Concrete subclasses must implement ``_copy``.
    """

    _user_journeys_df: pl.DataFrame
    _locations_gdf: gpd.GeoDataFrame

    def __init__(
        self, user_journeys_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, filters: list[str] | None = None
    ):
        """Initialise from the journey and location tables.

        Args:
            user_journeys_df: Journey legs conforming to ``USER_JOURNEY_SCHEMA``.
            locations_gdf: Locations conforming to ``LOCATIONS_SCHEMA``, sorted by ``loc_id``.
            filters: Optional list of ``type`` values to restrict locations (e.g. ``["zone"]``).
        """
        locations_gdf = locations_gdf.sort_values("loc_id")

        self._user_journeys_df = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
        self._locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

        self._type_filters = None if filters is None else filters.copy()
        self._aggregated_journeys = self._compute_agg_journeys(user_journeys_df)
        self._location_visits = self._compute_location_visits(self._aggregated_journeys)
        self._home_locations = self._compute_home_locations(self._location_visits)

    @cached_property
    def user_journeys_df(self) -> pl.DataFrame:
        """Journeys, filtered by active location filters on departure and arrival."""
        dep_filtered = self._filter_loc_types(self._user_journeys_df, loc_id_col="dep_loc_id", filter_users=True)
        arr_filtered = self._filter_loc_types(dep_filtered, loc_id_col="arr_loc_id")

        return arr_filtered

    @cached_property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        """Locations, filtered by active location filters."""
        return self._filter_loc_types(self._locations_gdf)

    @cached_property
    def locations_df(self) -> pl.DataFrame:
        """``locations_gdf`` as a polars DataFrame."""
        return utils.gdf_to_polars(self.locations_gdf)

    @cached_property
    def aggregated_journeys(self) -> pl.DataFrame:
        """Aggregated journeys, filtered by active location filters on departure and arrival."""
        filtered_trips = self._filter_loc_types(self._aggregated_journeys, loc_id_col="dep_loc_id", filter_users=True)
        filtered_trips = self._filter_loc_types(filtered_trips, loc_id_col="arr_loc_id")

        return filtered_trips

    @cached_property
    def location_visits(self) -> pl.DataFrame:
        """Location visits, filtered by active location filters."""
        return self._filter_loc_types(self._location_visits, filter_users=True)

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        """Home locations, filtered by active location filters."""
        return self._filter_loc_types(self._home_locations, filter_users=True)

    @cached_property
    def work_locations(self) -> pl.DataFrame:
        """Each user's main work locations."""
        return self.location_visits.filter(purpose=Purpose.WORK_MAIN).select("user_id", "loc_id").unique()

    @cached_property
    def edu_locations(self) -> pl.DataFrame:
        """Each user's study locations."""
        return self.location_visits.filter(purpose=Purpose.STUDY).select("user_id", "loc_id").unique()

    @cached_property
    def users_df(self) -> pl.DataFrame:
        """One row per user with home location, using ``user_id`` as the household id."""
        return (
            self
            ._filter_loc_types(self.home_locations)
            .sort("user_id")
            .select("user_id", hh_id="user_id", home_loc_id="loc_id")
        )

    @property
    def demographic_columns(self) -> list[str]:
        """Names of the demographic columns in ``users_df``; none by default."""
        return []

    @cached_property
    def user_ids(self) -> pl.Series:
        """Sorted user ids, filtered by active home location filters."""
        if self._type_filters is None:
            return self._home_locations["user_id"].unique().sort()

        valid_locations = utils.gdf_to_polars(self._locations_gdf).filter(pl.col("type").is_in(self._type_filters))
        valid_homes = self._home_locations.join(valid_locations.select("loc_id"), on="loc_id")

        users = valid_homes["user_id"].unique().sort()
        return users

    @cached_property
    def num_obs_days_per_user(self) -> pl.DataFrame:
        """Per-user observed-day count t_i, keyed by user_id."""
        return self.user_journeys_df.group_by("user_id").agg(n_days=pl.col("dep_day").n_unique().cast(pl.Int32))

    def with_filter(self, loc_types: str | list[str]) -> Self:
        """Return a new instance restricted to locations whose ``type`` is in ``loc_types``.

        Args:
            loc_types: One or more location type strings (e.g. ``"zone"``).

        Returns:
            A new ``SurveyData`` of the same concrete type with the filter applied.
        """
        filters = [loc_types] if isinstance(loc_types, str) else loc_types
        return self._copy(filters)

    def _filter_loc_types[TDataFrame: (pl.DataFrame, gpd.GeoDataFrame)](
        self, df: TDataFrame, loc_id_col: str = "loc_id", filter_users: bool = False
    ) -> TDataFrame:
        if self._type_filters is None:
            return df

        locations_df = utils.gdf_to_polars(self._locations_gdf)
        valid_loc_ids = locations_df.filter(pl.col("type").is_in(self._type_filters)).select(
            pl.col("loc_id").alias(loc_id_col)
        )

        if filter_users and "user_id" not in df.columns:
            raise ValueError(f"Cannot find column `user_id` in columns {df.columns}")
        elif filter_users and isinstance(df, pl.DataFrame):
            df = df.join(self.user_ids.to_frame(), on="user_id")
        elif filter_users and isinstance(df, gpd.GeoDataFrame):
            df = df.merge(self.user_ids.to_frame().to_pandas(), on="user_id")

        if isinstance(df, pl.DataFrame):
            return df.join(valid_loc_ids, on=loc_id_col)

        return df.merge(valid_loc_ids.to_pandas(), on=loc_id_col)

    @abstractmethod
    def _copy(self, filters: list[str] | None = None) -> Self: ...

    @staticmethod
    def _compute_agg_journeys(user_journeys_df: pl.DataFrame) -> pl.DataFrame:
        group_keys = ["user_id", "journey_id"]
        grouped_journeys = user_journeys_df.group_by(group_keys, maintain_order=True)

        modes = grouped_journeys.agg(modes="leg_mode")
        trips = grouped_journeys.agg(pl.all().gather([0, -1])).select(
            "user_id",
            "journey_id",
            (pl.col("leg_id").list.last() + 1).alias("num_legs"),
            pl.col("dep_day").list.first(),
            pl.col("dep_time").list.first(),
            pl.col("dep_purpose").list.first(),
            pl.col("dep_loc_id").list.first(),
            pl.col("arr_loc_id").list.last(),
            pl.col("arr_purpose").list.last(),
        )

        return trips.join(modes, on=group_keys)

    @staticmethod
    def _compute_location_visits(agg_journeys: pl.DataFrame) -> pl.DataFrame:
        departures = agg_journeys.select("user_id", purpose="dep_purpose", loc_id="dep_loc_id")
        arrivals = agg_journeys.select("user_id", purpose="arr_purpose", loc_id="arr_loc_id")
        all_visits = pl.concat([departures, arrivals])

        return all_visits.group_by("user_id", "purpose", "loc_id").agg(num_visits=pl.len()).sort("user_id")

    @staticmethod
    def _compute_home_locations(location_visits: pl.DataFrame) -> pl.DataFrame:
        """Computes home location from visits. Drops users with num_home_locations != 1."""
        locations_by_purpose = (
            location_visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")
        )
        home_locations = (
            locations_by_purpose
            .filter(purpose=Purpose.HOME)
            .filter(pl.col("loc_id").list.len() == 1)
            .with_columns(pl.col("loc_id").list.first())
            .drop("purpose")
        )

        return home_locations


class DataFrameStore:
    """Mixin that provides a standard cache-directory layout for concrete ``SurveyData`` subclasses."""

    @classmethod
    def _dirs(cls, cfg: DataConfig, project_root: Path | None = None, name: str | None = None) -> tuple[Path, Path]:
        project_root = project_root if project_root is not None else Path(".")
        suffix = "" if name is None else f"-{name}"
        data_dir = project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

        return project_root, data_dir


@dataclass(frozen=True)
class SurveyLoader[C: DataConfig, S]:
    """Bundle of the survey-specific steps for loading a survey and building its graph.

    Attributes:
        load: Loads the survey from the ``processed`` cache, builds and caches it on a cache miss.
        read_statistics: Reads the census statistics that enrich zone features.
        add_statistics: Adds population and job densities to the zone locations.
        build_extra_edges: Builds edges to add on top of the contiguity graph from its nodes (e.g. island zones); ``None`` adds none.
    """

    load: Callable[[C, Path | None], SurveyData]
    read_statistics: Callable[[C, Path | None], S]
    add_statistics: Callable[[gpd.GeoDataFrame, S], gpd.GeoDataFrame]
    build_extra_edges: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None


def build_special_locations() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame containing the single ``NA`` special location."""
    na_location = {
        "loc_id": "NA",
        "loc_name": "NA",
        "type": "na",
        "lon": NA_LON,
        "lat": NA_LAT,
    }

    return gpd.GeoDataFrame(
        [na_location],
        geometry=gpd.points_from_xy([NA_LON], [NA_LAT]),
        crs=CRS,
    )
