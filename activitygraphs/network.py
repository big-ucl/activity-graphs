"""NetworkData class that wraps raw DataFrames from different travel surveys into a unified interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Self, TypeVar

import geopandas as gpd
import polars as pl

from activitygraphs import utils
from activitygraphs.base import (
    CRS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    PTNodeType,
    Purpose,
)
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import check_schema

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"

type PTLayerBuilder = Callable[[pl.DataFrame, PTNodeType], tuple[pl.DataFrame, pl.DataFrame]]
type TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]
TDataFrame = TypeVar("TDataFrame", pl.DataFrame, gpd.GeoDataFrame)


class NetworkData(ABC):
    """Wrapper around survey DataFrames that provides a cached, filterable network view. Each datset (Geneva, Toronto, etc...)
    inherits from this class

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
        """
        Args:
            user_journeys_df: Journey legs conforming to ``USER_JOURNEY_SCHEMA``.
            locations_gdf: Locations conforming to ``LOCATIONS_SCHEMA``, sorted by ``loc_id``.
            filters: Optional list of ``type`` values to restrict locations (e.g. ``["subsector"]``).
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
        dep_filtered = self._filter_loc_types(self._user_journeys_df, loc_id_col="dep_loc_id", filter_users=True)
        arr_filtered = self._filter_loc_types(dep_filtered, loc_id_col="arr_loc_id")

        return arr_filtered

    @cached_property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        return self._filter_loc_types(self._locations_gdf)

    @cached_property
    def locations_df(self) -> pl.DataFrame:
        return utils.gdf_to_polars(self.locations_gdf)

    @cached_property
    def aggregated_journeys(self) -> pl.DataFrame:
        filtered_trips = self._filter_loc_types(self._aggregated_journeys, loc_id_col="dep_loc_id", filter_users=True)
        filtered_trips = self._filter_loc_types(filtered_trips, loc_id_col="arr_loc_id")

        return filtered_trips

    @cached_property
    def location_visits(self) -> pl.DataFrame:
        return self._filter_loc_types(self._location_visits, filter_users=True)

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        return self._filter_loc_types(self._home_locations, filter_users=True)

    @cached_property
    def work_locations(self) -> pl.DataFrame:
        return self.location_visits.filter(purpose=Purpose.WORK_MAIN).select("user_id", "loc_id").unique()

    @cached_property
    def edu_locations(self) -> pl.DataFrame:
        return self.location_visits.filter(purpose=Purpose.STUDY).select("user_id", "loc_id").unique()

    @cached_property
    def users_df(self) -> pl.DataFrame:
        return self._filter_loc_types(self.home_locations).sort("user_id").select("user_id", home_loc_id="loc_id")

    @cached_property
    def user_ids(self) -> pl.Series:
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
            loc_types: One or more location type strings (e.g. ``"subsector"``).

        Returns:
            A new ``NetworkData`` of the same concrete type with the filter applied.
        """
        filters = [loc_types] if isinstance(loc_types, str) else loc_types
        return self._copy(filters)

    def _filter_loc_types(self, df: TDataFrame, loc_id_col: str = "loc_id", filter_users: bool = False) -> TDataFrame:
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
        elif filter_users:
            df = df.merge(self.user_ids.to_frame().to_pandas(), on="user_id")

        if isinstance(df, pl.DataFrame):
            return df.join(valid_loc_ids, on=loc_id_col)

        return df.merge(valid_loc_ids.to_pandas(), on=loc_id_col)

    @abstractmethod
    def _copy(self, filters: list[str] | None = None): ...

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
        locations_by_purpose = (
            location_visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")
        )
        home_locations = (
            locations_by_purpose
            .filter(purpose=Purpose.HOME)
            .with_columns(pl.col("loc_id").list.first())
            .drop("purpose")
        )

        return home_locations


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
