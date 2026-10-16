"""OSRM-backed travel-time routers with optional in-memory caching."""

import abc
from collections.abc import Iterable
from typing import Self
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import polars as pl
import requests
from pypolyline.cutil import encode_coordinates

from activitygraphs.base import Mode
from activitygraphs.utils import check_schema

MAX_COORDS = 10_000
PRECISION = 5

MODE_TO_OSRM_PROFILE_MAP = {
    Mode.CAR: "car",
    Mode.MOTORCYCLE: "car",
    Mode.TAXI: "car",
    Mode.VEH_PASS: "car",
    Mode.WALK: "walk",
    Mode.CYCLE: "cycle",
}

EDGE_COORDS_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "orig_lon": pl.Float64,
    "orig_lat": pl.Float64,
    "dest_loc_id": pl.String,
    "dest_lon": pl.Float64,
    "dest_lat": pl.Float64,
})


class Router(abc.ABC):
    """Abstract base for travel-time routers that operate on edge coordinate DataFrames."""

    @abc.abstractmethod
    def travel_times(self, edge_coords_df: pl.DataFrame) -> pl.DataFrame:
        """Compute travel times for every (orig, dest) pair in ``edge_coords_df``.

        Args:
            edge_coords_df: DataFrame conforming to ``EDGE_COORDS_SCHEMA``.

        Returns:
            DataFrame with columns ``orig_loc_id``, ``dest_loc_id``, ``travel_time_min``.
        """

    def with_cache(self) -> Self:
        """Wrap this router in a ``CachedRouter`` for in-memory result caching."""
        return CachedRouter(self)


class TravelTimeCalculator:
    """Joins a ``Router`` with a location lookup to add ``travel_time_min`` to edge DataFrames."""

    def __init__(self, locations_gdf: gpd.GeoDataFrame, router: Router):
        check_schema(locations_gdf, {"loc_id": "object", "lon": "float64", "lat": "float64"}, ignore_extra_cols=True)

        self._locations_gdf = locations_gdf.copy()
        self._locations_df = pl.DataFrame(locations_gdf[["loc_id", "lon", "lat"]])
        self._origins = self._locations_df.select(pl.all().name.prefix("orig_"))
        self._destinations = self._locations_df.select(pl.all().name.prefix("dest_"))

        self._router = router

    def add_travel_times(self, edge_df: pl.DataFrame) -> pl.DataFrame:
        """Join ``travel_time_min`` onto ``edge_df`` using this instance's router.

        Args:
            edge_df: DataFrame with at least ``orig_loc_id`` and ``dest_loc_id`` columns.

        Returns:
            ``edge_df`` left-joined with a ``travel_time_min`` column (minutes, float).
        """
        check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)

        edge_locations = pl.concat([edge_df["orig_loc_id"], edge_df["dest_loc_id"]]).unique()
        if not edge_locations.is_in(self._locations_df["loc_id"]).all():
            raise ValueError("All locations in `edge_df` must exist in the router locations")

        edge_coords_df = (
            edge_df
            .select("orig_loc_id", "dest_loc_id")
            .join(self._origins, on="orig_loc_id")
            .join(self._destinations, on="dest_loc_id")
        )

        result = self._router.travel_times(edge_coords_df)
        check_schema(
            result, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String, "travel_time_min": pl.Float64})
        )

        return edge_df.join(result, on=["orig_loc_id", "dest_loc_id"], how="left")


class OSRMRouter(Router):
    """Router that calls a local OSRM HTTP table service."""

    def __init__(self, url: str, mode: Mode, max_coordinates: int = MAX_COORDS, precision: int = PRECISION):
        """Initialise the OSRM router.

        Args:
            url: Base URL of the OSRM server (e.g. ``"http://localhost:5000"``).
            mode: Travel mode; must be one of ``Mode.CAR``, ``Mode.WALK``, or ``Mode.CYCLE``.
            max_coordinates: Maximum number of unique locations per OSRM table request.
            precision: Polyline encoding precision (default 5).
        """
        if mode not in MODE_TO_OSRM_PROFILE_MAP.keys():
            raise ValueError(f"Invalid mode: {mode}, accepted: {MODE_TO_OSRM_PROFILE_MAP.keys()}")

        self.url = url.rstrip("/")
        self.mode = mode
        self.profile = MODE_TO_OSRM_PROFILE_MAP[mode]
        self.max_coordinates = max_coordinates
        self.precision = precision

    def travel_times(self, edge_coords_df: pl.DataFrame):
        """Return the OSRM travel time of each edge in ``edge_coords_df``."""
        check_schema(edge_coords_df, EDGE_COORDS_SCHEMA)

        locations = (
            pl
            .concat([
                edge_coords_df.select(loc_id="orig_loc_id", lon="orig_lon", lat="orig_lat"),
                edge_coords_df.select(loc_id="dest_loc_id", lon="dest_lon", lat="dest_lat"),
            ])
            .unique("loc_id")
            .sort("loc_id")
            .with_row_index()
        )

        coordinates = locations.select("lon", "lat").iter_rows()
        durations = self.table(coordinates)

        edge_indices = edge_coords_df.join(
            locations.select(orig_index="index", orig_loc_id="loc_id"), on="orig_loc_id"
        ).join(locations.select(dest_index="index", dest_loc_id="loc_id"), on="dest_loc_id")

        edge_indices_np = edge_indices.select("orig_index", "dest_index").to_numpy()
        travel_times = durations[edge_indices_np[:, 0], edge_indices_np[:, 1]]

        return edge_coords_df.select("orig_loc_id", "dest_loc_id").with_columns(travel_time_min=travel_times)

    def table(self, coordinates: Iterable[tuple[float, float]]) -> np.ndarray:
        """Call the OSRM ``/table`` endpoint and return a duration matrix in minutes.

        Args:
            coordinates: (lon, lat) pairs; length must not exceed ``max_coordinates``.

        Returns:
            2-D numpy array of shape ``[n, n]`` with travel times in minutes (OSRM returns seconds).
        """
        coordinates = list(coordinates)

        if len(coordinates) > self.max_coordinates:
            raise ValueError(f"Too many coordinates: {len(coordinates)}, maximum: {self.max_coordinates}")

        polyline = encode_coordinates(coordinates, self.precision)
        polyline = quote(polyline, safe="")
        full_url = f"{self.url}/table/v1/{self.profile}/polyline({polyline})"
        response = requests.get(full_url)

        if not response.ok:
            raise ConnectionError(
                f"OSRM routing service error: status code={response.status_code}, content={response.content}"
            )

        json = response.json()
        if json["code"] != "Ok":
            raise ConnectionError(f"OSRM routing error:\nstatus code={json['code']}\nmessage={json['message']}")

        travel_times_min = np.array(json["durations"]) / 60
        return travel_times_min


class CachedRouter(Router):
    """Decorator that adds an in-memory (orig, dest) → travel_time_min cache to any ``Router``."""

    def __init__(self, router: Router):
        self.router = router
        self.cache: dict[tuple[str, str], float] = {}

    def travel_times(self, edge_coords_df: pl.DataFrame) -> pl.DataFrame:
        """Return the travel time of each edge, querying the wrapped router only for uncached edges."""
        check_schema(edge_coords_df, EDGE_COORDS_SCHEMA)

        cache_keys = pl.Series(self.cache.keys(), dtype=pl.List(pl.String)).implode()

        is_edge_in_cache = pl.concat_list("orig_loc_id", "dest_loc_id").is_in(cache_keys)
        cache_hits = edge_coords_df.filter(is_edge_in_cache)
        cache_misses = edge_coords_df.filter(~is_edge_in_cache)

        cache_misses_tt = self.router.travel_times(cache_misses)
        cache_hits_tt = cache_hits.select(
            "orig_loc_id",
            "dest_loc_id",
            travel_time_min=pl
            .concat_list("orig_loc_id", "dest_loc_id")
            .replace(self.cache)
            .list.item()
            .cast(pl.Float64),
        )

        return pl.concat([cache_hits_tt, cache_misses_tt])
