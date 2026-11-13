"""Build the per-user PyG tensors from a survey and corresp. graph."""

from collections.abc import Collection
from pathlib import Path
from typing import cast

import city2graph as c2g
import geopandas as gpd
import numpy as np
import polars as pl
import torch
import torch_geometric as pyg

from activitygraphs.base import Purpose
from activitygraphs.config import Config, DataConfig
from activitygraphs.graph import load_graph
from activitygraphs.graph.access import ACCESS_SCHEMA
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, TensorSchema, infer_schema
from activitygraphs.surveys.base import SurveyData
from activitygraphs.surveys.registry import load_survey
from activitygraphs.tensors.bundles import LocationTensors, SurveyTensors, UserTensors, check_tensors
from activitygraphs.tensors.cache import read_tensors, save_tensors, tensors_path
from activitygraphs.utils import get_project_root

COLS_EXCLUDED_FROM_FEATURES = [
    "loc_name",
    "type",
    "lon",
    "lat",
    "is_work",
    "is_edu",
    "is_visited",
    "purpose",
    "geometry",
    "original_geometry",
]

USER_NODE_SCHEMA = TensorSchema("user_nodes", (ColumnSpec("is_home", ColumnKind.BINARY),))
EDGE_SCHEMA = TensorSchema("edges", (ColumnSpec("weight", ColumnKind.COUNT),))
DISTANCE_SCHEMA = TensorSchema("distances", (ColumnSpec("home_distance", ColumnKind.COUNT),))
TARGET_SCHEMA = TensorSchema("targets", (ColumnSpec("is_visited", ColumnKind.BINARY),))


# =========================================
# Home nodes and visit labels
# =========================================


def _build_home_nodes(survey: SurveyData) -> torch.Tensor:
    """Build the home node of each user, the position of the user's home location in the node order.

    Returns:
        Int64 tensor of shape ``[n_users]``.
    """
    user_to_home_loc = dict(survey.home_locations.iter_rows())
    location_to_node = {loc_id: node_idx for node_idx, loc_id in enumerate(survey.locations_df["loc_id"])}
    home_nodes = survey.user_ids.replace_strict(user_to_home_loc).replace_strict(
        location_to_node, return_dtype=pl.Int64
    )

    return torch.tensor(home_nodes.to_numpy(), dtype=torch.long)


def _build_visit_labels(survey: SurveyData, purposes: Collection[Purpose]) -> torch.Tensor:
    """Build each user's visit labels over all graph nodes, from the visits made for one of ``purposes``.

    Args:
        survey: The survey whose users and locations index the labels.
        purposes: The visit purposes that count as a visit.

    Returns:
        Float tensor of shape ``[n_users, n_nodes]``, one at each node the user visited for one of ``purposes``.
    """
    visits = survey.location_visits.filter(pl.col("purpose").cast(pl.String).is_in([str(p) for p in purposes]))

    return _build_user_node_indicator(survey, visits, "is_visited")


def _build_user_node_indicator(survey: SurveyData, indicator_df: pl.DataFrame, col_name: str) -> torch.Tensor:
    """Build a per-user indicator over all graph nodes, one where ``[user_id, loc_id]`` appears in ``indicator_df``.

    Returns:
        Float tensor of shape ``[n_users, n_nodes]``.
    """
    locations = survey.locations_df.select("loc_id").with_row_index("loc_order")
    indicators = (
        survey.user_ids
        .to_frame()
        .join(locations, how="cross")
        .lazy()
        .pipe(_add_indicator_column, indicator_df, col_name)
        .sort("user_id", "loc_order")
        .collect()
    )

    return torch.tensor(indicators[col_name].to_numpy(), dtype=torch.float32).reshape(
        len(survey.user_ids), len(locations)
    )


def _build_access(nodes_gdf: gpd.GeoDataFrame) -> torch.Tensor:
    """Return the ``ACCESS_SCHEMA`` columns of the nodes, in node order, ``[n_nodes, n_access_features]``."""
    access_values = nodes_gdf[ACCESS_SCHEMA.names].to_numpy(dtype=np.float32)
    return torch.from_numpy(access_values)


def _build_node_distances(nodes_gdf: gpd.GeoDataFrame) -> torch.Tensor:
    """Build the distance between every pair of graph nodes.

    Uses Euclidean distance between centroids. Row ``i`` holds the distances from node ``i`` to every node.

    Returns:
        Float tensor of shape ``[n_nodes, n_nodes]``.
    """
    nodes = nodes_gdf.sort_index().to_crs(nodes_gdf.estimate_utm_crs())
    coords = nodes.geometry.get_coordinates()

    n_nodes = len(coords)
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for node_idx, loc_id in enumerate(nodes_gdf.index):
        distances[node_idx, :] = np.linalg.norm(coords - coords.loc[loc_id], axis=1)

    return torch.from_numpy(distances)


def _add_indicator_column(feature_df: pl.LazyFrame, indicator_df: pl.DataFrame, col_name: str):
    """Add an indicator column with name ``col_name`` to ``feature_df`` which is ``True`` if ``[loc_id, user_id]`` appears in ``indicator_df`` and ``False`` otherwise."""
    true_indicators = indicator_df.select("user_id", "loc_id", pl.lit(True).alias(col_name)).unique().lazy()

    return feature_df.join(true_indicators, on=["user_id", "loc_id"], how="left").with_columns(
        pl.col(col_name).fill_null(False)
    )


# =========================================
# User attributes and household IDs
# =========================================


def _build_user_attributes(survey: SurveyData) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the user attribute tensors from SurveyData, split into numeric columns and categorical codes.

    Columns follow ``survey.demographics_schema``. If the survey has none (e.g. TPG), both tensors have zero columns.

    Returns:
        Tuple ``(numeric, codes)``: the float32 numeric columns ``[n_users, n_numeric_attr]`` and the int64 codes of
        the categorical columns ``[n_users, n_categorical_attr]``, each in schema column order.
    """
    demo_schema = survey.demographics_schema
    n_users = len(survey.user_ids)

    numeric_names = demo_schema.numeric.names
    if numeric_names:
        numeric_values = survey.users_df.select(numeric_names).to_numpy()
        numeric_tensor = torch.tensor(numeric_values, dtype=torch.float32)
    else:
        numeric_tensor = torch.empty((n_users, 0), dtype=torch.float32)

    categorical_tensor = torch.empty((n_users, 0), dtype=torch.long)
    for spec in demo_schema.categorical.columns:
        column_codes = _encode_categories(survey.users_df[spec.name], spec)
        categorical_tensor = torch.cat([categorical_tensor, column_codes.unsqueeze(1)], dim=1)

    return numeric_tensor, categorical_tensor


def _encode_categories(values: pl.Series, spec: ColumnSpec) -> torch.Tensor:
    """Encode the values of a categorical column as codes, following the categories of ``spec``.

    The values are compared to the categories as strings. ``spec.categories[i]`` gets code ``i + 1``. A missing value
    or a value outside the categories gets code 0.

    Returns:
        Int64 tensor of shape ``[len(values)]``.
    """
    codes = {category: code for code, category in enumerate(spec.categories, start=1)}
    as_strings = values.cast(pl.String)
    encoded = as_strings.replace_strict(codes, default=0, return_dtype=pl.Int64).fill_null(0)

    return torch.tensor(encoded.to_numpy(), dtype=torch.long)


def _build_group_ids(survey: SurveyData) -> torch.Tensor:
    """Create a tensor of numeric ``group_id``s matching the ``hh_id`` household IDs, shape ``[n_users]``."""
    group_ids = survey.users_df.select("hh_id").unique(maintain_order=True).with_row_index(name="group_id")
    user_groups = survey.users_df.select("user_id", "hh_id").join(group_ids, on="hh_id", how="left")
    assert user_groups["user_id"].equals(survey.user_ids)

    return user_groups.select("group_id").to_torch().long().squeeze()


# =========================================
# PyG / Torch creation
# =========================================


def build_tensors(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> SurveyTensors:
    """Convert the graph and the survey into tensor form.

    Returns:
        Tuple ``(locations, users, schema)``: the tensors of the shared graph, the per-user tensors and the dataset
        schema.

    Raises:
        ValueError: The graph has no edge index, or the tensors fail ``check_tensors``.
    """
    excluded_cols = [*COLS_EXCLUDED_FROM_FEATURES, *ACCESS_SCHEMA.names]
    node_feature_cols = [col for col in nodes_gdf.columns if col not in excluded_cols]

    pyg_graph = c2g.gdf_to_pyg(
        nodes_gdf,
        edges_gdf,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=EDGE_SCHEMA.names,
        keep_geom=False,
        device="cpu",
    )

    access = _build_access(nodes_gdf)
    distance_m = _build_node_distances(nodes_gdf)
    node_ids = _build_node_ids(survey, nodes_gdf)
    locations = _build_location_tensors(cast(pyg.data.Data, pyg_graph), access, distance_m, node_ids)

    attributes, attributes_cat = _build_user_attributes(survey)
    users = UserTensors(
        home_node=_build_home_nodes(survey),
        visited=_build_visit_labels(survey, list(Purpose)),
        attributes=attributes,
        attributes_cat=attributes_cat,
        group_ids=_build_group_ids(survey),
        user_ids=_build_user_ids(survey),
    )

    schema = _build_schema(survey, nodes_gdf, node_feature_cols)
    check_tensors(locations, users, schema)

    return locations, users, schema


def load_tensors(cfg: Config, project_root: Path | None = None) -> SurveyTensors:
    """Return the ``SurveyTensors`` cached at ``tensors_path``, building and caching them first if needed.

    Raises:
        ValueError: The cached tensors fail ``check_tensors``.
    """
    project_root = get_project_root(project_root)
    root = tensors_path(cfg.data, project_root)

    try:
        return read_tensors(root)
    except FileNotFoundError:
        tensors = _build_survey_tensors(cfg.data, project_root)
        save_tensors(root, *tensors)
        return tensors


def _build_user_ids(survey: SurveyData) -> pl.DataFrame:
    """Return the survey ``user_id`` and ``hh_id`` of each user, in tensor row order."""
    user_ids = survey.users_df.select("user_id", "hh_id")
    assert user_ids["user_id"].equals(survey.user_ids)

    return user_ids


def _build_node_ids(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Return the survey ``loc_id`` of each graph node, in node order.

    Raises:
        ValueError: The graph's nodes are not the survey locations in ``loc_id`` order, the order of the per-user
            tensors.
    """
    node_ids = pl.DataFrame({"loc_id": nodes_gdf.index.to_list()})
    if not node_ids["loc_id"].equals(survey.locations_df["loc_id"]):
        raise ValueError("The graph's nodes are not the survey locations in `loc_id` order.")

    return node_ids


def _build_schema(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, node_feature_cols: list[str]) -> DatasetSchema:
    """Return the final dataset schema. Extracts schema from ``survey`` and infers schema on ``nodes_gdf``.

    Args:
        survey: The survey which is the source of the demographics schema.
        nodes_gdf: The graph nodes that contain the location feature columns.
        node_feature_cols: The columns names of the location features, in tensor column order.

    Returns:
        The dataset schema.
    """
    locations_schema = infer_schema("locations", nodes_gdf, node_feature_cols, {})

    return DatasetSchema(
        locations=locations_schema,
        access=ACCESS_SCHEMA,
        edges=EDGE_SCHEMA,
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=survey.demographics_schema,
        distances=DISTANCE_SCHEMA,
        targets=TARGET_SCHEMA,
    )


def _build_location_tensors(
    pyg_graph: pyg.data.Data, access: torch.Tensor, distance_m: torch.Tensor, node_ids: pl.DataFrame
) -> LocationTensors:
    """Return the location tensors of the PyG graph built by city2graph.

    Raises:
        ValueError: The graph has no edge index.
    """
    if pyg_graph.edge_index is None:
        raise ValueError("The graph has no edge index.")

    return LocationTensors(
        x=cast(torch.Tensor, pyg_graph.x),
        access=access,
        edge_index=pyg_graph.edge_index,
        edge_attr=cast(torch.Tensor, pyg_graph.edge_attr),
        distance_m=distance_m,
        node_ids=node_ids,
    )


def _build_survey_tensors(cfg: DataConfig, project_root: Path) -> SurveyTensors:
    """Build the SurveyTensors from the (cached or rebuilt) ``SurveyData`` and survey Graph described by ``cfg``."""
    survey = load_survey(cfg, project_root).with_filter("zone")
    nodes_gdf, edges_gdf = load_graph(survey, cfg, project_root)

    return build_tensors(survey, nodes_gdf, edges_gdf)
