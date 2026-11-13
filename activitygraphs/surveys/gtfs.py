"""GTFS reader: the feeds of a dataset, read into one set of tables with every id prefixed by its feed."""

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import polars as pl

from activitygraphs.config import GTFSFeeds


class PTMode(StrEnum):
    """Public transport mode of a GTFS route."""

    METRO = "metro"
    RAIL = "rail"
    TRAM = "tram"
    BUS = "bus"
    FERRY = "ferry"
    OTHER = "other"


PT_MODE_DTYPE = pl.Enum([mode.value for mode in PTMode])

BASIC_ROUTE_TYPE_MODES = {
    0: PTMode.TRAM,
    1: PTMode.METRO,
    2: PTMode.RAIL,
    3: PTMode.BUS,
    4: PTMode.FERRY,
    5: PTMode.OTHER,
    6: PTMode.OTHER,
    7: PTMode.OTHER,
    11: PTMode.BUS,
    12: PTMode.METRO,
}

# Hundreds prefix of extended route type (route_type // 100), see https://developers.google.com/transit/gtfs/reference/extended-route-types
EXTENDED_ROUTE_TYPE_MODES_BY_HUNDRED = {
    1: PTMode.RAIL,
    2: PTMode.BUS,
    3: PTMode.RAIL,
    4: PTMode.METRO,
    5: PTMode.METRO,
    6: PTMode.METRO,
    7: PTMode.BUS,
    8: PTMode.BUS,
    9: PTMode.TRAM,
    10: PTMode.FERRY,
    11: PTMode.OTHER,
    12: PTMode.FERRY,
    13: PTMode.OTHER,
    14: PTMode.OTHER,
    15: PTMode.OTHER,
    16: PTMode.OTHER,
    17: PTMode.OTHER,
}

ROUTE_TYPE_MODES = {
    **{route_type: mode.value for route_type, mode in BASIC_ROUTE_TYPE_MODES.items()},
    **{route_type: EXTENDED_ROUTE_TYPE_MODES_BY_HUNDRED[route_type // 100].value for route_type in range(100, 1800)},
}
"""Maps GTFS basic and extended route types to ``PTMode`` enum"""

ID_COLUMNS = {"stop_id", "parent_station", "route_id", "trip_id", "service_id"}
TIME_COLUMNS = {"arrival_time", "departure_time"}
DATE_COLUMNS = {"start_date", "end_date", "date"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

TABLE_COLUMNS: dict[str, dict[str, pl.DataType]] = {
    "stops": {
        "stop_id": pl.String(),
        "stop_name": pl.String(),
        "stop_lat": pl.Float64(),
        "stop_lon": pl.Float64(),
        "location_type": pl.Int8(),
        "parent_station": pl.String(),
    },
    "routes": {
        "route_id": pl.String(),
        "route_short_name": pl.String(),
        "route_long_name": pl.String(),
        "route_type": pl.Int32(),
    },
    "trips": {"trip_id": pl.String(), "route_id": pl.String(), "service_id": pl.String()},
    "stop_times": {
        "trip_id": pl.String(),
        "stop_id": pl.String(),
        "stop_sequence": pl.Int32(),
        "arrival_time": pl.Int32(),
        "departure_time": pl.Int32(),
    },
    "calendar": {
        "service_id": pl.String(),
        **{weekday: pl.Boolean() for weekday in WEEKDAYS},
        "start_date": pl.Date(),
        "end_date": pl.Date(),
    },
    "calendar_dates": {"service_id": pl.String(), "date": pl.Date(), "exception_type": pl.Int8()},
}
OPTIONAL_COLUMNS = {
    "stops": {"location_type", "parent_station"},
    "routes": {"route_short_name", "route_long_name"},
}
OPTIONAL_TABLES = {"calendar", "calendar_dates"}


@dataclass(frozen=True)
class GTFSTables:
    """GTFS tables of the feeds of one dataset, and the service date read from them.

    Every table has a ``feed`` column. Each id column is prefixed by its feed name and a colon.

    Attributes:
        service_date: The weekday on which the timetables are read.
        stops_df: One row per stop, station or entrance.
        routes_df: One row per route, with its ``mode`` from its route type.
        trips_df: One row per trip.
        stop_times_df: One row per call of a trip at a stop. Times are seconds after midnight of the
            service day, and can exceed 24 hours.
        calendar_df: The weekly services and their date ranges.
        calendar_dates_df: The services added (``exception_type`` 1) or removed (2) on single dates.
    """

    service_date: date
    stops_df: pl.DataFrame
    routes_df: pl.DataFrame
    trips_df: pl.LazyFrame
    stop_times_df: pl.LazyFrame
    calendar_df: pl.DataFrame
    calendar_dates_df: pl.DataFrame

    def trips_on_service_date(self) -> pl.LazyFrame:
        """Return the trips that run on ``service_date``."""
        weekday = WEEKDAYS[self.service_date.weekday()]
        in_range = (pl.col("start_date") <= self.service_date) & (pl.col("end_date") >= self.service_date)
        weekly = self.calendar_df.filter(in_range & pl.col(weekday)).select("service_id")

        exceptions = self.calendar_dates_df.filter(pl.col("date") == self.service_date)
        added = exceptions.filter(pl.col("exception_type") == 1).select("service_id")
        removed = exceptions.filter(pl.col("exception_type") == 2).select("service_id")

        services = pl.concat([weekly, added]).unique().join(removed, on="service_id", how="anti")
        return self.trips_df.join(services.lazy(), on="service_id")


def read_gtfs(gtfs_feeds: GTFSFeeds, gtfs_dir: Path) -> GTFSTables:
    """Read and combine all GTFS feeds of a dataset into one set of ``GTFSTables``. A feed is a directory or a zip file.

    Args:
        gtfs_feeds: The list of paths to the feeds within ``gtfs_dir`` and the desired service date.
        gtfs_dir: Root directory of the GTFS feed paths.

    Returns:
        The concatenated tables of all the feeds.

    Raises:
        ValueError: If a feed lacks a required column or has a route type outside the GTFS route types.
    """
    all_tables: dict[str, list[pl.LazyFrame]] = {table: [] for table in TABLE_COLUMNS}

    for feed_name, path in gtfs_feeds.feeds.items():
        feed_path = gtfs_dir / path
        feed_tables = _read_feed(feed_name, feed_path).items()

        for table_name, table_frame in feed_tables:
            all_tables[table_name].append(table_frame)

    combined_tables = {table: pl.concat(frames) for table, frames in all_tables.items()}

    route_mode_to_enum_expr = pl.col("route_type").replace_strict(ROUTE_TYPE_MODES, return_dtype=PT_MODE_DTYPE)
    try:
        routes_df = combined_tables["routes"].with_columns(mode=route_mode_to_enum_expr).collect()
    except pl.exceptions.InvalidOperationError as error:
        raise ValueError(f"route type outside the GTFS route types: {error}") from error

    return GTFSTables(
        service_date=date.fromisoformat(gtfs_feeds.service_date),
        stops_df=combined_tables["stops"].collect(),
        routes_df=routes_df,
        trips_df=combined_tables["trips"],
        stop_times_df=combined_tables["stop_times"],
        calendar_df=combined_tables["calendar"].collect(),
        calendar_dates_df=combined_tables["calendar_dates"].collect(),
    )


def _read_feed(feed_name: str, feed_path: Path) -> dict[str, pl.LazyFrame]:
    """Read the GTFS tables of one feed (directory or zip file) with typed columns and prefixed ids."""
    table_reader = _build_dir_reader(feed_path) if feed_path.is_dir() else _build_archive_reader(feed_path)

    tables = {}
    for table_name, table_columns in TABLE_COLUMNS.items():
        table_source = table_reader(table_name)

        if table_source is None and table_name not in OPTIONAL_TABLES:
            raise FileNotFoundError(f"{feed_path} has no {table_name} table")
        if table_source is None:
            tables[table_name] = pl.LazyFrame(schema={"feed": pl.String(), **table_columns})
            continue

        raw_table = pl.scan_csv(table_source, infer_schema=False).rename(lambda name: name.strip().lstrip("﻿"))
        typed_table = _typed_table(raw_table, feed_name, table_name, table_columns)

        tables[table_name] = typed_table

    return tables


def _build_archive_reader(path: Path) -> Callable[[str], Path | io.BytesIO | None]:
    """Return a function that reads a feed table from a ``.zip`` GTFS feed."""
    archive = zipfile.ZipFile(path)
    members = {Path(name).name: name for name in archive.namelist()}

    def _reader(table_name: str) -> io.BytesIO | None:
        file_name = f"{table_name}.txt"
        return io.BytesIO(archive.read(members[file_name])) if file_name in members else None

    return _reader


def _build_dir_reader(path: Path) -> Callable[[str], Path | io.BytesIO | None]:
    """Return a function that reads a feed table from  GTFS feed."""

    def _reader(table_name: str) -> Path | None:
        file_path = path / f"{table_name}.txt"
        return file_path if file_path.exists() else None

    return _reader


def _typed_table(
    raw_table: pl.LazyFrame, feed_name: str, table: str, required_columns: dict[str, pl.DataType]
) -> pl.LazyFrame:
    """Select and type the columns of a raw table, padding the absent optional columns with nulls."""
    present_cols = set(raw_table.collect_schema().names())
    optional_cols = OPTIONAL_COLUMNS.get(table, set())

    selected_cols = [pl.lit(feed_name).alias("feed")]
    for col_name, dtype in required_columns.items():
        if col_name not in present_cols and col_name not in optional_cols:
            raise ValueError(f"{feed_name} {table}.txt has no column {col_name}")
        if col_name not in present_cols:
            selected_cols.append(pl.lit(None, dtype=dtype).alias(col_name))
            continue

        stripped_col_expr = pl.col(col_name).str.strip_chars()
        empty_stripped_to_null_expr = pl.when(stripped_col_expr != "").then(stripped_col_expr)
        cast_col = _cast(empty_stripped_to_null_expr, col_name, dtype, feed_name)

        selected_cols.append(cast_col.alias(col_name))

    return raw_table.select(selected_cols)


def _cast(expr: pl.Expr, col_name: str, dtype: pl.DataType, feed: str) -> pl.Expr:
    """Convert a stripped string column to a GTFS type."""
    if col_name in ID_COLUMNS:
        return pl.lit(f"{feed}:") + expr
    if col_name in TIME_COLUMNS:  # Convert manually since GTFS times can go beyond 24 hrs.
        parts = expr.str.split(":").list.eval(pl.element().cast(pl.Int32))
        return parts.list.get(0) * 3600 + parts.list.get(1) * 60 + parts.list.get(2)
    if col_name in DATE_COLUMNS:
        return expr.str.to_date("%Y%m%d")
    if dtype == pl.Boolean():
        return expr.cast(pl.Int8).cast(pl.Boolean)

    return expr.cast(dtype)
