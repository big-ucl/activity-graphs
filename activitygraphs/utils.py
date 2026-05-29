"""Schema validation, geometry helpers, and I/O utilities."""

from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from typing import Hashable, TypeVar

import geopandas as gpd
import pandas as pd
import polars as pl
import torch

from activitygraphs.base import CRS, EDGE_LIST_SCHEMA, LOCATIONS_SCHEMA
from activitygraphs.config import DataConfig

PandasSchema = Mapping[str, str]
TDataFrame = TypeVar("TDataFrame", gpd.GeoDataFrame, pl.DataFrame)
TSchema = TypeVar("TSchema", pl.Schema, PandasSchema)


def get_project_root(project_root: Path | None = None) -> Path:
    """Return ``project_root`` if provided, otherwise ``Path(".")``."""
    project_root: Path = project_root if project_root is not None else Path(".")
    return project_root


def check_schema(df: TDataFrame, schema: TSchema, ignore_extra_cols: bool = False) -> TDataFrame:
    """Checks that a DataFrame matches provided Schema.

    Args:
        df (pl.DataFrame | gpd.GeoDataFrame): the dataframe to be checked
        schema (pl.Schema | dict): the schema to check against
        ignore_extra_cols (bool): ignore extra columns not in the schema, defaults to False

    Raises:
        ValueError: if the DataFrame does not match the schema

    Returns:
        pl.DataFrame: the valid DataFrame
    """

    if isinstance(df, pl.DataFrame) and isinstance(schema, pl.Schema):
        return check_polars_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    if (isinstance(df, gpd.GeoDataFrame) or isinstance(df, pd.DataFrame)) and isinstance(schema, dict):
        return check_geopandas_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    raise ValueError(f"Invalid types: Got {type(df)=}, {type(schema)=}")


def check_polars_schema(df: pl.DataFrame, schema: pl.Schema, ignore_extra_cols: bool) -> pl.DataFrame:
    """Validate a Polars DataFrame against a ``pl.Schema``, raise ``ValueError`` on mismatch."""
    df_items = set(df.schema.items())
    schema_items = set(schema.items())

    missing = schema_items - df_items
    extra = df_items - schema_items

    if missing or (extra and not ignore_extra_cols):
        raise ValueError(
            f"Schemas do not match:\nMissing elements: {missing}\n"
            + (f"Extra elements: {extra}" if not ignore_extra_cols else "")
        )

    return df


def check_geopandas_schema(gdf: gpd.GeoDataFrame, schema: PandasSchema, ignore_extra_cols: bool) -> gpd.GeoDataFrame:
    """Validate a GeoPandas/Pandas DataFrame against a column-name -> dtype-string mapping."""

    def dtypes_match(dtype: str, expected: str) -> bool:
        return (dtype == "str" and expected == "object") or dtype == expected

    if not ignore_extra_cols and set(schema.keys()) != set(gdf.columns):
        raise ValueError(
            f"Columns do not match schema. \nGot: {sorted(gdf.columns)}\nExpected: {sorted(schema.keys())}"
        )

    if ignore_extra_cols and not set(schema.keys()).issubset(set(gdf.columns)):
        missing = set(schema.keys()).difference(set(gdf.columns))

        raise ValueError(f"Columns do not match schema. \nGot: {sorted(schema.keys())}\nMissing columns: {missing}")

    # noinspection PyTypeChecker
    mismatches = [
        (col, dtype, schema[col]) for col, dtype in gdf.dtypes.items() if col in schema and dtypes_match(dtype, schema)
    ]
    if mismatches:
        errors = [f"\n\t`{col}`: got {dtype}, expected {expected}" for col, dtype, expected in mismatches]
        raise ValueError("Invalid types: " + "".join(errors))

    return gdf


def check_geometry_shapes(geometry: gpd.GeoSeries, *shapes: str) -> gpd.GeoSeries:
    """Raise ``ValueError`` if any geometry in ``geometry`` is not one of the permitted ``shapes``."""
    geom_types: pd.Series = geometry.geom_type
    is_valid = geom_types.isin(shapes)

    if not is_valid.all():
        invalid = list(geom_types[~is_valid].unique())
        raise ValueError(f"Invalid shapes in geometry. Permitted: {shapes}, found {invalid}")

    return geometry


def check_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Checks that a Tensor has a given shape, ignores size of dimension if size is set to -1.

    Args:
        tensor (torch.Tensor): the tensor
        shape (tuple[int, ...]): a tuple of dimension sizes

    Raises:
        ValueError: if the shape is invalid

    Returns:
        torch.Tensor: the checked tensor
    """
    if len(tensor.shape) != len(shape):
        raise ValueError(f"Tensor has invalid number of dims, got shape {tensor.shape}, expected {shape}")

    for i, (t, s) in enumerate(zip(tensor.shape, shape)):
        if t != s and s != -1:
            raise ValueError(
                f"Tensor has invalid shape along dimension {i}, got shape {tensor.shape}, expected {shape}"
            )

    return tensor


def gdf_to_polars(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Convert a GeoDataFrame to a Polars DataFrame, dropping the geometry column."""
    return pl.DataFrame(gdf.drop(columns=["geometry"]))


def convert_locations_to_point_geometry(locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace a locations GeoDataFrame's geometry with point geometries derived from ``lon``/``lat``."""
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    locations_gdf = locations_gdf.copy()
    locations_gdf.set_geometry(gpd.points_from_xy(locations_gdf["lon"], locations_gdf["lat"], crs=CRS), inplace=True)

    return locations_gdf


def extract_unique_loc_ids(*edge_dfs: TDataFrame) -> list[str]:
    """Collect all unique location IDs from the ``orig_loc_id`` and ``dest_loc_id`` columns of one or more edge DataFrames."""
    loc_dfs = []
    for edge_df in edge_dfs:
        edge_df = edge_df if isinstance(edge_df, pl.DataFrame) else gdf_to_polars(edge_df)
        edge_df = check_schema(edge_df, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        loc_dfs.append(edge_df["orig_loc_id"])
        loc_dfs.append(edge_df["dest_loc_id"])

    return pl.concat(loc_dfs).unique().to_list()


def convert_excel_to_parquet(data_path: Path, *files: Path) -> list[Path]:
    """Convert Excel files under ``data_path`` to parquet in-place and return the new file paths."""
    new_files = []

    for file in files:
        df = pl.read_excel(data_path / file)

        new_file = file.with_suffix(".parquet")
        df.write_parquet(data_path / new_file)
        new_files.append(new_file)

    return new_files


def read_from_parquet(path: Path, schema: dict = None) -> pl.DataFrame:
    """Read a parquet file into a Polars DataFrame with optional schema overrides."""
    schema = {} if schema is None else schema

    df = pl.read_parquet(path)
    return pl.DataFrame(df, schema_overrides=schema)


def add_lon_lat_from_centroid(
    gdf: gpd.GeoDataFrame, index_col: str, lon_name="lon", lat_name="lat"
) -> gpd.GeoDataFrame:
    """Add ``lon``/``lat`` columns computed from polygon centroids (projected to WGS-84)."""
    gdf = gdf.copy()

    projected_crs = gdf.estimate_utm_crs()
    centroids = gdf.to_crs(projected_crs).set_index(index_col).centroid.to_crs(CRS)
    centroids = gpd.GeoDataFrame(centroids, columns=["centroid"])

    gdf = gdf.join(centroids, on=index_col)
    gdf[lon_name] = gdf["centroid"].x
    gdf[lat_name] = gdf["centroid"].y

    return gdf.drop(columns=["centroid"])


class DataFrameStore(ABC):
    """Mixin that provides a standard cache-directory layout for concrete ``NetworkData`` subclasses."""

    @classmethod
    def _dirs(cls, cfg: DataConfig, project_root: Path | None = None, name: str | None = None) -> tuple[Path, Path]:
        project_root: Path = project_root if project_root is not None else Path(".")
        suffix = "" if name is None else f"-{name}"
        data_dir = project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

        return project_root, data_dir


def invert_mapping(mapping: dict[Hashable, list[Hashable]]) -> dict[Hashable, Hashable]:
    """Invert a one-to-many mapping to a many-to-one mapping; raises ``ValueError`` on duplicate values."""
    inversion = {}

    for k, vs in mapping.items():
        for v in vs:
            if v in inversion:
                raise ValueError(f"Duplicate values in mapping for {k=}: v1={v} and v2={inversion[v]}")

            inversion[v] = k

    return inversion
