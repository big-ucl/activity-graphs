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

# Single source of truth for the per-user node feature columns, in the order they are
# appended after the graph's node features in the concatenated node feature matrix `x`.
# `is_home` must stay first so that the derived home-column index (see VisitDataset)
# remains valid for `is_home_node_mask`, the conditional baseline and the home exclusion.
USER_NODE_FEATURE_NAMES = ["is_home"]


@dataclass(frozen=True)
class SurveyTensors:
    """The shared PyG graph and per-user tensors of a survey, with the survey ids of the users and nodes.

    Attributes:
        pyg_graph: The shared PyG graph, with node features ``x``, ``edge_index`` and ``edge_attr``.
        user_node_features: Per-user node features, ``[n_users, n_nodes, n_feat]``.
        visit_labels: Per-user visit labels, ``[n_users, n_nodes, 1]``.
        user_attributes: Per-user attributes, ``[n_users, n_attr]``.
        distances: Per-user distance from home, ``[n_users, n_nodes, 1]``.
        group_ids: Per-user group/hh-ids, ``[n_users]``.
        user_ids: Survey ``user_id`` and ``hh_id`` of each user, one row per tensor row.
        node_ids: Survey ``loc_id`` of each node, one row per graph node.
    """

    pyg_graph: pyg.data.Data
    user_node_features: torch.Tensor
    visit_labels: torch.Tensor
    user_attributes: torch.Tensor
    distances: torch.Tensor
    group_ids: torch.Tensor
    user_ids: pl.DataFrame
    node_ids: pl.DataFrame


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


def build_user_attributes(survey: SurveyData) -> torch.Tensor:
    """Create the user attribute float32 tensor of shape ``[n_users, n_attr]`` from SurveyData.

    Columns follow ``survey.demographics_schema``. If the survey has none (e.g. TPG), returns a (n_users, 1) dummy
    tensor of ones.
    """
    col_names = survey.demographics_schema.names
    if not col_names:
        return torch.ones((len(survey.user_ids), 1), dtype=torch.float32)

    user_attributes = survey.users_df.select(col_names)

    return torch.tensor(user_attributes.to_numpy(), dtype=torch.float32)


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


def build_tensors(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> SurveyTensors:
    """Convert the graph and the survey into PyG and tensor form.

    Returns:
        The shared PyG graph, the per-user tensors and the survey ids of the users and nodes.
    """
    user_node_features = build_home_indicator(survey)
    visit_labels = build_visit_labels(survey, list(Purpose))
    user_attributes = build_user_attributes(survey)
    distances = build_home_distances(survey, nodes_gdf)
    hh_ids = build_group_ids(survey)

    node_feature_cols = [col for col in nodes_gdf.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    pyg_graph = c2g.gdf_to_pyg(
        nodes_gdf,
        edges_gdf,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=["weight"],
        keep_geom=False,
        device="cpu",
    )

    return SurveyTensors(
        pyg_graph=cast(pyg.data.Data, pyg_graph),
        user_node_features=user_node_features,
        visit_labels=visit_labels,
        user_attributes=user_attributes,
        distances=distances,
        group_ids=hh_ids,
        user_ids=build_user_ids(survey),
        node_ids=build_node_ids(survey, nodes_gdf),
    )
