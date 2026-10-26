"""Build the per-user PyG tensors from a survey and corresp. graph."""

from dataclasses import dataclass
from typing import cast

import city2graph as c2g
import geopandas as gpd
import numpy as np
import polars as pl
import torch
import torch_geometric as pyg

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

# Single source of truth for the per-user spatial feature columns, in the order they are
# appended after the graph's node features in the concatenated node feature matrix `x`.
# `is_home` must stay first so that the derived home-column index (see ActivityDataset)
# remains valid for `is_home_node_mask`, the conditional baseline and the home exclusion.
SPATIAL_FEATURE_NAMES = ["is_home"]


@dataclass(frozen=True)
class SurveyTensors:
    """Bundle of tensors used by ML pipeline: the (shared) PyG graph and the (per-user) tensors for a survey.

    Attributes:
        pyg_graph: The shared PyG graph, with node features ``x``, ``edge_index`` and ``edge_attr``.
        spatial_features: Per-user spatial features, ``[n_users, n_nodes, n_feat]``.
        spatial_labels: Per-user labels, ``[n_users, n_nodes, 1]``.
        demographics: Individual demographics, ``[n_users, n_demo]``.
        distances: Per-user distance from home, ``[n_users, n_nodes, 1]``.
        group_ids: Per-user group/hh-ids, ``[n_users]``.
    """

    pyg_graph: pyg.data.Data
    spatial_features: torch.Tensor
    spatial_labels: torch.Tensor
    demographics: torch.Tensor
    distances: torch.Tensor
    group_ids: torch.Tensor


# =========================================
# Spatial demographics
# =========================================


def add_user_cols(
    user_id: str,
    nodes_gdf: gpd.GeoDataFrame,
    location_visits: pl.DataFrame,
    home_locations: pl.DataFrame,
    work_locations: pl.DataFrame,
    edu_locations: pl.DataFrame,
) -> gpd.GeoDataFrame:
    """Annotate graph nodes with per-user binary indicators and visit purposes.

    Adds columns ``is_home``, ``is_work``, ``is_edu``, ``is_visited``, and ``purpose``
    (comma-separated string of visit purposes) to a copy of ``nodes_gdf``.
    """
    nodes_gdf = nodes_gdf.copy().reset_index()

    user_visits = location_visits.filter(user_id=user_id)["loc_id"].to_list()
    home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
    work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
    edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()

    nodes_gdf["is_home"] = nodes_gdf["loc_id"].isin(home_location).astype(int)
    nodes_gdf["is_work"] = nodes_gdf["loc_id"].isin(work_location).astype(int)
    nodes_gdf["is_edu"] = nodes_gdf["loc_id"].isin(edu_location).astype(int)
    nodes_gdf["is_visited"] = nodes_gdf["loc_id"].isin(user_visits).astype(int)

    visit_purposes = (
        location_visits
        .filter(user_id=user_id)
        .select("loc_id", pl.col("purpose"))
        .group_by("loc_id")
        .agg(pl.col("purpose").str.join(", "))
        .to_pandas()
    )
    nodes_gdf = nodes_gdf.merge(visit_purposes, on="loc_id", how="left")
    nodes_gdf = nodes_gdf.set_index("loc_id").sort_index()

    return nodes_gdf


def build_spatial_demographics(survey: SurveyData) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-user spatial feature and label tensors over all graph nodes.

    Returns:
        Tuple ``(spatial_features, spatial_labels)`` of shapes
        ``[n_users, n_nodes, n_features]`` and ``[n_users, n_nodes, 1]``.
        Features contain ``is_home``; labels contain ``is_visited``.
    """
    locations = survey.locations_df.select("loc_id").with_row_index("loc_order")
    all_users_and_locs = survey.user_ids.to_frame().join(locations, how="cross").lazy()

    n_users = len(survey.user_ids)
    n_locs = len(locations)

    # Build spatial features (i.e. will be appended to PyG data.x)
    feature_df = (
        all_users_and_locs
        .pipe(add_indicator_column, survey.home_locations, "is_home")
        .pipe(add_indicator_column, survey.work_locations, "is_work")
        .pipe(add_indicator_column, survey.edu_locations, "is_edu")
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    feature_cols = SPATIAL_FEATURE_NAMES
    spatial_features = torch.tensor(feature_df.select(feature_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(feature_cols)
    )

    # Build spatial demographics labels (i.e. will form PyG data.y)
    label_df = (
        all_users_and_locs
        .pipe(add_indicator_column, survey.location_visits, "is_visited")
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    label_cols = ["is_visited"]
    spatial_labels = torch.tensor(label_df.select(label_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(label_cols)
    )

    return spatial_features, spatial_labels


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
# Individual demographics and household IDs
# =========================================


def build_individual_demographics(survey: SurveyData) -> torch.Tensor:
    """Create the demographic float32 tensor of shape ``[n_users, n_demo_features]`` from SurveyData.

    Columns follow ``survey.demographic_columns``. If there are none (e.g. TPG), returns a (n_users, 1) dummy tensor
    of ones.
    """
    if len(survey.demographic_columns) == 0:
        return torch.ones((len(survey.user_ids), 1), dtype=torch.float32)

    indiv_demographics = survey.users_df.select(survey.demographic_columns)

    return torch.tensor(indiv_demographics.to_numpy(), dtype=torch.float32)


def build_group_ids(survey: SurveyData) -> torch.Tensor:
    """Create a tensor of numeric ``group_id``s matching the ``hh_id`` household IDs, shape ``[n_users]``."""
    group_ids = survey.users_df.select("hh_id").unique(maintain_order=True).with_row_index(name="group_id")
    user_groups = survey.users_df.select("user_id", "hh_id").join(group_ids, on="hh_id", how="left")
    assert user_groups["user_id"].equals(survey.user_ids)

    return user_groups.select("group_id").to_torch().long().squeeze()


# =========================================
# PyG / Torch creation
# =========================================


def build_tensors(survey: SurveyData, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> SurveyTensors:
    """Convert the graph and the survey into PyG and tensor form.

    Returns:
        The shared PyG graph and the per-user tensors.
    """
    spatial_features, spatial_labels = build_spatial_demographics(survey)
    demographics = build_individual_demographics(survey)
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
        spatial_features=spatial_features,
        spatial_labels=spatial_labels,
        demographics=demographics,
        distances=distances,
        group_ids=hh_ids,
    )
