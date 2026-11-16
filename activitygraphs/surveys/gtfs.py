"""GTFS reader: the feeds of a dataset, read into one set of tables with every id prefixed by its feed."""

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import polars as pl
import shapely

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

SUBSET_KEY_COLUMNS = {
    "stop_id": "stop",
    "parent_station": "stop",
    "from_stop_id": "stop",
    "to_stop_id": "stop",
    "trip_id": "trip",
    "from_trip_id": "trip",
    "to_trip_id": "trip",
    "route_id": "route",
    "from_route_id": "route",
    "to_route_id": "route",
    "service_id": "service",
    "shape_id": "shape",
}
"""Id columns of the GTFS tables by type of GTFS ``_id``. Used by ``write_feed_subset`` to filter out discarded items."""


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


def write_feed_subsets(
    gtfs_feeds: GTFSFeeds, gtfs_dir: Path, clip_polygon: shapely.Geometry, out_dir: Path
) -> list[Path]:
    """Write each feed as a "clipped" zipped GTFS feed: holding only the trips on the service date calling inside ``clip_polygon``.

    This function discards trips that: 1) do not run on the service date or 2) do not call inside ``clip_polygon``.
    All stops, routes, services, shapes, and calendar dates that are no longer mentioned in the remaining trips are
    also discarded.

    Remaining trips include all calls, including the calls outside ``clip_polygon``. If a stop is kept, the
    corresponding `parent_stop` is also kept.

    Args:
        gtfs_feeds: The paths of the feeds within ``gtfs_dir`` and the service date.
        gtfs_dir: Root directory of the GTFS feed paths.
        clip_polygon: The area of the trips to keep, in longitude and latitude.
        out_dir: Directory of the zipped feeds, one ``{feed name}.zip`` per feed.

    Returns:
        The path of each non-empty zipped feed, in the order of ``gtfs_feeds.feeds``.
    """
    gtfs = read_gtfs(gtfs_feeds, gtfs_dir)
    subset_ids = _subset_ids(gtfs, clip_polygon)
    service_date = gtfs.service_date.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    feed_paths = []
    for feed_name, path in gtfs_feeds.feeds.items():
        feed_prefix = f"{feed_name}:"
        feed_ids = {}
        for kind, ids in subset_ids.items():
            feed_ids[kind] = ids.filter(ids.str.starts_with(feed_prefix)).str.strip_prefix(feed_prefix)

        if feed_ids["trip"].is_empty():
            continue

        feed_path = out_dir / f"{feed_name}.zip"
        _write_feed_subset(gtfs_dir / path, feed_ids, service_date, feed_path)
        feed_paths.append(feed_path)

    return feed_paths


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

        raw_table = pl.scan_csv(table_source, infer_schema=False).rename(lambda name: name.lstrip("\ufeff").strip())
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


def _list_feed_tables(path: Path) -> list[str]:
    """Return the name of every table of a feed (directory or zip file), without the ``.txt`` extension."""
    if path.is_dir():
        return sorted(file.stem for file in path.glob("*.txt"))

    with zipfile.ZipFile(path) as archive:
        return sorted(Path(name).stem for name in archive.namelist() if name.endswith(".txt"))


def _subset_ids(gtfs: GTFSTables, clip_polygon: shapely.Geometry) -> dict[str, pl.Series]:
    """Return the prefixed ids kept by ``write_feed_subsets``, keyed by kind: ``stop``, ``trip``, ``route``, ``service``."""
    stop_lon = gtfs.stops_df["stop_lon"].to_numpy()
    stop_lat = gtfs.stops_df["stop_lat"].to_numpy()
    inside_mask = shapely.contains_xy(clip_polygon, stop_lon, stop_lat)
    inside_stops = gtfs.stops_df.filter(pl.Series(inside_mask)).select("stop_id")
    inside_calls = gtfs.stop_times_df.join(inside_stops.lazy(), on="stop_id")

    service_trips = gtfs.trips_on_service_date()
    kept_trip_ids = service_trips.join(inside_calls, on="trip_id").select("trip_id").unique()
    kept_trips = service_trips.join(kept_trip_ids, on="trip_id").collect()

    called_stops = gtfs.stop_times_df.join(kept_trip_ids, on="trip_id").select("stop_id").unique().collect()
    parent_stations = gtfs.stops_df.join(called_stops, on="stop_id").select(stop_id="parent_station").drop_nulls()
    kept_stops = pl.concat([called_stops, parent_stations]).unique()

    return {
        "stop": kept_stops["stop_id"],
        "trip": kept_trips["trip_id"],
        "route": kept_trips["route_id"].unique(),
        "service": kept_trips["service_id"].unique(),
    }


def _write_feed_subset(feed_path: Path, feed_ids: dict[str, pl.Series], service_date: str, out_path: Path) -> None:
    """Write the rows of every table of one feed whose ids are in ``feed_ids`` to the zip file ``out_path``.

    A table left with no row is not written. Bytes that are not valid UTF-8 are replaced. Fields beyond the header of
    a table are dropped.
    """
    table_reader = _build_dir_reader(feed_path) if feed_path.is_dir() else _build_archive_reader(feed_path)
    table_names = _list_feed_tables(feed_path)
    ordered_names = sorted(table_names, key=lambda name: name != "trips")
    subset_ids = dict(feed_ids)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for table_name in ordered_names:
            table_source = table_reader(table_name)
            if table_source is None:
                raise FileNotFoundError(f"{feed_path} has no {table_name} table")

            raw_table = pl.scan_csv(table_source, infer_schema=False, encoding="utf8-lossy", truncate_ragged_lines=True)
            raw_table = raw_table.rename(lambda name: name.lstrip("\ufeff").strip())  # Strip chars in column names

            table_subset = _table_subset(raw_table, subset_ids)

            if table_name == "calendar_dates":
                table_subset = table_subset.filter(pl.col("date").str.strip_chars() == service_date)

            subset_rows = table_subset.collect()
            if table_name == "trips" and "shape_id" in subset_rows.columns:
                subset_ids["shape"] = subset_rows["shape_id"].str.strip_chars().drop_nulls().unique()

            if subset_rows.height > 0:
                archive.writestr(f"{table_name}.txt", subset_rows.write_csv())


def _table_subset(raw_table: pl.LazyFrame, subset_ids: dict[str, pl.Series]) -> pl.LazyFrame:
    """Filter ``raw_table`` to keep only rows with 1) empty id columns or 2) an id in ``subset_ids``."""
    table_cols = raw_table.collect_schema().names()

    for col_name in table_cols:
        kind = SUBSET_KEY_COLUMNS.get(col_name)
        if kind not in subset_ids:
            continue

        stripped_col_expr = pl.col(col_name).str.strip_chars()
        empty_expr = stripped_col_expr.is_null() | (stripped_col_expr == "")
        raw_table = raw_table.filter(empty_expr | stripped_col_expr.is_in(subset_ids[kind].implode()))

    return raw_table


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
