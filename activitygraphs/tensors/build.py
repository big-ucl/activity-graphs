"""Build the per-user PyG tensors from a survey and corresp. graph."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import cast

import city2graph as c2g
import geopandas as gpd
import numpy as np
import polars as pl
import torch
import torch_geometric as pyg

from activitygraphs.base import Purpose
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, TensorSchema, infer_schema
from activitygraphs.surveys.base import SurveyData

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


@dataclass(frozen=True)
class SurveyTensors:
    """The shared PyG graph and per-user tensors of a survey, with the survey ids of the users and nodes.

    Attributes:
        pyg_graph: The shared PyG graph, with node features ``x``, ``edge_index`` and ``edge_attr``.
        user_node_features: Per-user node features, ``[n_users, n_nodes, n_feat]``.
        visit_labels: Per-user visit labels, ``[n_users, n_nodes, 1]``.
        user_attributes: Per-user numeric attributes, ``[n_users, n_numeric_attr]``.
        user_attributes_cat: Per-user categorical attribute codes, int64, ``[n_users, n_categorical_attr]``.
        distances: Per-user distance from home, ``[n_users, n_nodes, 1]``.
        group_ids: Per-user group/hh-ids, ``[n_users]``.
        user_ids: Survey ``user_id`` and ``hh_id`` of each user, one row per tensor row.
        node_ids: Survey ``loc_id`` of each node, one row per graph node.
        schema: Column schema of each tensor group.

    Raises:
        ValueError: The graph has no edge index, the tensors and ids disagree on the number of users or nodes, a
            group's tensors have a different number of numeric or categorical columns than its schema, or a user
            does not have exactly one home node.
    """

    pyg_graph: pyg.data.Data
    user_node_features: torch.Tensor
    visit_labels: torch.Tensor
    user_attributes: torch.Tensor
    user_attributes_cat: torch.Tensor
    distances: torch.Tensor
    group_ids: torch.Tensor
    user_ids: pl.DataFrame
    node_ids: pl.DataFrame
    schema: DatasetSchema

    def __post_init__(self) -> None:
        """Check that the tensors agree on the number of users and nodes, that their widths match the schema, and that
        every user has exactly one home node."""
        if self.pyg_graph.edge_index is None:
            raise ValueError("The graph has no edge index.")

        _require_equal_counts(
            "User",
            {
                "user_node_features": self.user_node_features.shape[0],
                "visit_labels": self.visit_labels.shape[0],
                "user_attributes": self.user_attributes.shape[0],
                "user_attributes_cat": self.user_attributes_cat.shape[0],
                "distances": self.distances.shape[0],
                "group_ids": self.group_ids.shape[0],
                "user_ids": self.user_ids.height,
            },
        )
        _require_equal_counts(
            "Node",
            {
                "pyg_graph": self.pyg_graph.num_nodes,
                "user_node_features": self.user_node_features.shape[1],
                "visit_labels": self.visit_labels.shape[1],
                "distances": self.distances.shape[1],
                "node_ids": self.node_ids.height,
            },
        )

        numeric_and_categorical_widths = {
            "locations": (self.pyg_graph.num_node_features, 0),
            "edges": (self.pyg_graph.num_edge_features, 0),
            "user_nodes": (self.user_node_features.shape[-1], 0),
            "user_attributes": (self.user_attributes.shape[-1], self.user_attributes_cat.shape[-1]),
            "distances": (self.distances.shape[-1], 0),
            "targets": (self.visit_labels.shape[-1], 0),
        }

        for schema in self.schema.groups:
            widths = numeric_and_categorical_widths[schema.group]
            schema_widths = (schema.numeric.width, schema.categorical.width)
            if widths != schema_widths:
                raise ValueError(
                    f"The `{schema.group}` tensors have (numeric, categorical) widths {widths} but its schema has "
                    f"{schema_widths}."
                )

        is_home_idx = self.schema.user_nodes.index_of("is_home")
        home_counts = self.user_node_features[:, :, is_home_idx].sum(dim=1)
        num_bad = int((home_counts != 1).sum())
        if num_bad:
            raise ValueError(f"{num_bad} users do not have exactly one home node.")


def _require_equal_counts(count_name: str, counts: dict[str, int | None]) -> None:
    """Raise unless every count in ``counts`` is the same.

    Args:
        count_name: The display name of the count that is being checked (e.g. "User").
        counts: The count read from each tensor or id frame, keyed by the tensor/frame name.

    Raises:
        ValueError: The counts differ.
    """
    if len(set(counts.values())) > 1:
        listed = ", ".join(f"{name} {count}" for name, count in counts.items())
        raise ValueError(f"{count_name} count mismatch: {listed}.")


# =========================================
# User node features and visit labels
# =========================================


def build_home_indicator(survey: SurveyData) -> torch.Tensor:
    """Build each user's home indicator over all graph nodes.

    Returns:
        Float tensor of shape ``[n_users, n_nodes, 1]``, one at the user's home node.
    """
    return build_user_node_indicator(survey, survey.home_locations, "is_home")


def build_visit_labels(survey: SurveyData, purposes: Collection[Purpose]) -> torch.Tensor:
    """Build each user's visit labels over all graph nodes, from the visits made for one of ``purposes``.

    Args:
        survey: The survey whose users and locations index the labels.
        purposes: The visit purposes that count as a visit.

    Returns:
        Float tensor of shape ``[n_users, n_nodes, 1]``, one at each node the user visited for one of ``purposes``.
    """
    visits = survey.location_visits.filter(pl.col("purpose").cast(pl.String).is_in([str(p) for p in purposes]))

    return build_user_node_indicator(survey, visits, "is_visited")


def build_user_node_indicator(survey: SurveyData, indicator_df: pl.DataFrame, col_name: str) -> torch.Tensor:
    """Build a per-user indicator over all graph nodes, one where ``[user_id, loc_id]`` appears in ``indicator_df``.

    Returns:
        Float tensor of shape ``[n_users, n_nodes, 1]``.
    """
    locations = survey.locations_df.select("loc_id").with_row_index("loc_order")
    indicators = (
        survey.user_ids
        .to_frame()
        .join(locations, how="cross")
        .lazy()
        .pipe(add_indicator_column, indicator_df, col_name)
        .sort("user_id", "loc_order")
        .collect()
    )

    return torch.tensor(indicators[col_name].to_numpy(), dtype=torch.float32).reshape(
        len(survey.user_ids), len(locations), 1
    )


def build_home_distances(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame) -> torch.Tensor:
    """Build a distance-from-home tensor over all graph nodes for every user.

    Uses Euclidean distance between centroids.

    Returns:
        Float tensor of shape ``[n_users, n_nodes, 1]``.
    """
    nodes = nodes_gdf.sort_index().to_crs(nodes_gdf.estimate_utm_crs())
    coords = nodes.geometry.get_coordinates()

    home_by_user = dict(survey.home_locations.iter_rows())

    n_users, n_nodes = len(survey.user_ids), len(coords)
    distances = np.zeros((n_users, n_nodes, 1), dtype=np.float32)

    for user_idx, user_id in enumerate(survey.user_ids):
        home_loc_id = home_by_user[user_id]
        distances[user_idx, :, 0] = np.linalg.norm(coords - coords.loc[home_loc_id], axis=1)

    return torch.from_numpy(distances)


def add_indicator_column(feature_df: pl.LazyFrame, indicator_df: pl.DataFrame, col_name: str):
    """Add an indicator column with name ``col_name`` to ``feature_df`` which is ``True`` if ``[loc_id, user_id]`` appears in ``indicator_df`` and ``False`` otherwise."""
    true_indicators = indicator_df.select("user_id", "loc_id", pl.lit(True).alias(col_name)).unique().lazy()

    return feature_df.join(true_indicators, on=["user_id", "loc_id"], how="left").with_columns(
        pl.col(col_name).fill_null(False)
    )


# =========================================
# User attributes and household IDs
# =========================================


def build_user_attributes(survey: SurveyData) -> tuple[torch.Tensor, torch.Tensor]:
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
        column_codes = encode_categories(survey.users_df[spec.name], spec)
        categorical_tensor = torch.cat([categorical_tensor, column_codes.unsqueeze(1)], dim=1)

    return numeric_tensor, categorical_tensor


def encode_categories(values: pl.Series, spec: ColumnSpec) -> torch.Tensor:
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


def build_group_ids(survey: SurveyData) -> torch.Tensor:
    """Create a tensor of numeric ``group_id``s matching the ``hh_id`` household IDs, shape ``[n_users]``."""
    group_ids = survey.users_df.select("hh_id").unique(maintain_order=True).with_row_index(name="group_id")
    user_groups = survey.users_df.select("user_id", "hh_id").join(group_ids, on="hh_id", how="left")
    assert user_groups["user_id"].equals(survey.user_ids)

    return user_groups.select("group_id").to_torch().long().squeeze()


# =========================================
# PyG / Torch creation
# =========================================


def build_user_ids(survey: SurveyData) -> pl.DataFrame:
    """Return the survey ``user_id`` and ``hh_id`` of each user, in tensor row order."""
    user_ids = survey.users_df.select("user_id", "hh_id")
    assert user_ids["user_id"].equals(survey.user_ids)

    return user_ids


def build_node_ids(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Return the survey ``loc_id`` of each graph node, in node order.

    Raises:
        ValueError: The graph's nodes are not the survey locations in ``loc_id`` order, the order of the per-user
            tensors.
    """
    node_ids = pl.DataFrame({"loc_id": nodes_gdf.index.to_list()})
    if not node_ids["loc_id"].equals(survey.locations_df["loc_id"]):
        raise ValueError("The graph's nodes are not the survey locations in `loc_id` order.")

    return node_ids


def build_schema(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, node_feature_cols: list[str]) -> DatasetSchema:
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
        edges=EDGE_SCHEMA,
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=survey.demographics_schema,
        distances=DISTANCE_SCHEMA,
        targets=TARGET_SCHEMA,
    )


def build_tensors(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> SurveyTensors:
    """Convert the graph and the survey into PyG and tensor form.

    Returns:
        Bundle containing the shared PyG graph, the per-user tensors, the survey ids of the users and nodes,
        and the dataset schema.
    """
    user_node_features = build_home_indicator(survey)
    visit_labels = build_visit_labels(survey, list(Purpose))
    user_attributes, user_attributes_cat = build_user_attributes(survey)
    distances = build_home_distances(survey, nodes_gdf)
    hh_ids = build_group_ids(survey)

    node_feature_cols = [col for col in nodes_gdf.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    pyg_graph = c2g.gdf_to_pyg(
        nodes_gdf,
        edges_gdf,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=EDGE_SCHEMA.names,
        keep_geom=False,
        device="cpu",
    )

    return SurveyTensors(
        pyg_graph=cast(pyg.data.Data, pyg_graph),
        user_node_features=user_node_features,
        visit_labels=visit_labels,
        user_attributes=user_attributes,
        user_attributes_cat=user_attributes_cat,
        distances=distances,
        group_ids=hh_ids,
        user_ids=build_user_ids(survey),
        node_ids=build_node_ids(survey, nodes_gdf),
        schema=build_schema(survey, nodes_gdf, node_feature_cols),
    )
