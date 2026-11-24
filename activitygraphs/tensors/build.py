"""Build the per-user PyG tensors from a survey and corresp. graph."""

from collections.abc import Collection
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch_geometric.utils import to_undirected

from activitygraphs.base import Purpose
from activitygraphs.config import Config, DataConfig
from activitygraphs.graph import GraphTables, load_graph
from activitygraphs.graph.access import ACCESS_SCHEMA
from activitygraphs.graph.rail import EDGE_INDEX_NAMES, RAIL_EDGE_SCHEMA
from activitygraphs.graph.travel_times import TRAVEL_TIME_SCHEMA
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, EdgeKind, TensorSchema, infer_schema
from activitygraphs.surveys.base import SurveyData
from activitygraphs.surveys.registry import load_survey
from activitygraphs.tensors.bundles import (
    OPPORTUNITY_COLUMNS,
    EdgeTensors,
    LocationTensors,
    SurveyTensors,
    UserTensors,
    check_tensors,
)
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
CONTIGUITY_EDGE_SCHEMA = TensorSchema("contiguity_edges", (ColumnSpec("weight", ColumnKind.COUNT),))
DISTANCE_SCHEMA = TensorSchema("distances", (ColumnSpec("home_distance", ColumnKind.COUNT),))
TARGET_SCHEMA = TensorSchema("targets", (ColumnSpec("is_visited", ColumnKind.BINARY),))
EDGE_SCHEMAS = {EdgeKind.CONTIGUITY: CONTIGUITY_EDGE_SCHEMA, EdgeKind.RAIL: RAIL_EDGE_SCHEMA}

AREA_COLUMN = "area"


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


def _build_opportunities(nodes_gdf: gpd.GeoDataFrame) -> torch.Tensor:
    """Return the number of jobs and of residents of each node, following ``OPPORTUNITY_COLUMNS``.

    Each number is the density of the node times the area of the node.

    Returns:
        Float32 tensor of shape ``[n_nodes, n_opportunity_columns]``.
    """
    densities = nodes_gdf[list(OPPORTUNITY_COLUMNS)].to_numpy(dtype=np.float32)
    areas = nodes_gdf[[AREA_COLUMN]].to_numpy(dtype=np.float32)

    return torch.from_numpy(densities * areas)


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


def _build_edges(edges: pd.DataFrame, node_ids: pl.DataFrame, schema: TensorSchema) -> EdgeTensors:
    """Return the edges of one kind as node indices in both directions, with their ``schema`` columns.

    Args:
        edges: Edges between the zones, indexed by (``source``, ``target``) ``loc_id``, one row per pair of zones.
        node_ids: Survey ``loc_id`` of each node, in node order.
        schema: The columns of the edge features, in tensor column order.

    Returns:
        The edges, ``[2, 2 * num_edges]``, sorted by source node and then by target node, and their features,
        ``[2 * num_edges, schema.width]``.

    Raises:
        ValueError: ``edges`` holds a pair of zones in more than one row, or a zone paired with itself.
    """
    num_nodes = node_ids.height
    node_index = pd.Series(np.arange(num_nodes), index=node_ids["loc_id"].to_list())
    source_ids = edges.index.get_level_values("source")
    target_ids = edges.index.get_level_values("target")
    source = node_index.loc[source_ids].to_numpy()
    target = node_index.loc[target_ids].to_numpy()
    one_way_index = torch.from_numpy(np.stack([source, target]))
    one_way_values = torch.tensor(edges[schema.names].to_numpy(dtype=np.float32))
    edge_index, edge_values = to_undirected(one_way_index, one_way_values, num_nodes=num_nodes)

    if edge_index.shape[1] != 2 * len(edges):
        raise ValueError(f"The `{schema.group}` table holds a pair of zones in more than one row, or a self loop.")

    return EdgeTensors(edge_index, edge_values)


def _build_location_features(nodes_gdf: gpd.GeoDataFrame, node_feature_cols: list[str]) -> torch.Tensor:
    """Return the location features of every node, in node order, ``[n_nodes, n_location_features]``."""
    feature_values = nodes_gdf[node_feature_cols].to_numpy().astype(np.float32)
    return torch.from_numpy(feature_values)


def _build_travel_times(travel_times: pd.DataFrame, node_ids: pl.DataFrame) -> torch.Tensor:
    """Return a tensor of travel times between every ordered pair of nodes, following ``TRAVEL_TIME_SCHEMA``.

    Args:
        travel_times: Travel times between every ordered pair of zones, indexed by (``source``, ``target``) ``loc_id``.
        node_ids: Survey ``loc_id`` of each node, in node order.

    Returns:
        Float tensor of shape ``[n_nodes, n_nodes, n_travel_time_features]``. Row ``i`` holds the travel times from
        node ``i``.
    """
    loc_ids = node_ids["loc_id"].to_list()
    pair_index = pd.MultiIndex.from_product([loc_ids, loc_ids], names=EDGE_INDEX_NAMES)
    pair_values = travel_times.reindex(pair_index)[TRAVEL_TIME_SCHEMA.names].to_numpy(dtype=np.float32)
    num_nodes = len(loc_ids)

    return torch.from_numpy(pair_values).reshape(num_nodes, num_nodes, TRAVEL_TIME_SCHEMA.width)


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


def build_tensors(survey: SurveyData, graph: GraphTables, excluded_location_features: Collection[str]) -> SurveyTensors:
    """Convert the graph and the survey into tensor form.

    Args:
        survey: The survey.
        graph: The nodes, the edges of every kind and the travel times of the graph.
        excluded_location_features: Columns of the graph nodes left out of the location features.

    Returns:
        Tuple ``(locations, users, schema)``: the tensors of the shared graph, the per-user tensors and the dataset
        schema.

    Raises:
        KeyError: A column of ``excluded_location_features`` is not a column of the graph nodes.
        ValueError: The tensors fail ``check_tensors``.
    """
    nodes_gdf = graph.nodes
    missing_cols = [col for col in excluded_location_features if col not in nodes_gdf.columns]
    if missing_cols:
        raise KeyError(f"The graph nodes have no columns {missing_cols} to leave out of the location features.")

    excluded_cols = [*COLS_EXCLUDED_FROM_FEATURES, *ACCESS_SCHEMA.names, *excluded_location_features]
    node_feature_cols = [col for col in nodes_gdf.columns if col not in excluded_cols]

    node_ids = _build_node_ids(survey, nodes_gdf)
    edges = {kind: _build_edges(graph.edges[kind], node_ids, EDGE_SCHEMAS[kind]) for kind in EdgeKind}
    locations = LocationTensors(
        x=_build_location_features(nodes_gdf, node_feature_cols),
        access=_build_access(nodes_gdf),
        opportunities=_build_opportunities(nodes_gdf),
        edges=edges,
        distance_m=_build_node_distances(nodes_gdf),
        travel_times_min=_build_travel_times(graph.travel_times, node_ids),
        node_ids=node_ids,
    )

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
        edges=EDGE_SCHEMAS,
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=survey.demographics_schema,
        distances=DISTANCE_SCHEMA,
        travel_times=TRAVEL_TIME_SCHEMA,
        targets=TARGET_SCHEMA,
    )


def _build_survey_tensors(cfg: DataConfig, project_root: Path) -> SurveyTensors:
    """Build the SurveyTensors from the (cached or rebuilt) ``SurveyData`` and survey Graph described by ``cfg``."""
    survey = load_survey(cfg, project_root).with_filter("zone")
    graph = load_graph(survey, cfg, project_root)

    return build_tensors(survey, graph, cfg.excluded_location_features)
