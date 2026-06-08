"""Build Network Graphs and per-user PyG tensors from NetworkData."""

import pickle
from pathlib import Path
from typing import Callable, cast

import city2graph as c2g
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from joblib import Parallel, delayed


from activitygraphs.base import CRS
from activitygraphs.config import (
    DataConfig,
    GenevaStatsInputs,
    StatsInputs,
    TorontoStatsInputs,
    TorontoDataConfig,
    GenevaDataConfig,
)
from activitygraphs.data.geneva import GenevaData
from activitygraphs.data.overture import Overture
from activitygraphs.data.statistics import (
    add_geneva_population_job_statistics,
    add_toronto_population_job_statistics,
)
from activitygraphs.data.toronto import TorontoData
from activitygraphs.network import NetworkData
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

# Single source of truth for the per-user spatial feature columns, in the order they are
# appended after the network node features in the concatenated node feature matrix `x`.
# `is_home` must stay first so that the derived home-column index (see ActivityDataset)
# remains valid for `extract_is_home` / the conditional baseline.
SPATIAL_FEATURE_NAMES = ["is_home"]

# =========================================
# A. Network graph
# =========================================

type NetworkGraphBuilder = Callable[
    [gpd.GeoDataFrame, Overture, StatsInputs], tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
]


def load_network_graph(
    data: NetworkData,
    cfg: DataConfig,
    build_network_graph: NetworkGraphBuilder,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the Network Graph from cache, or build and cache it.

    Cached parquet files are stored under ``cfg.paths.processed/<name>/``.

    Args:
        data: ``NetworkData`` instance for this dataset.
        cfg: Dataset config providing processed and raw paths.
        build_network_graph: Callable ``(locations_gdf, overture, stats_cfg) -> (nodes, edges)``.
        project_root: Repo root, defaults to ``Path(".")``.
        name: Sub-directory name for the cached files.

    Returns:
        Tuple of ``(nodes_gdf, edges_gdf)`` GeoDataFrames.
    """
    root = get_project_root(project_root)
    network_path = root / cfg.paths.processed / name
    nodes_path = network_path / "nodes.parquet"
    edges_path = network_path / "edges.parquet"

    if nodes_path.exists() and edges_path.exists():
        network_nodes = gpd.read_parquet(nodes_path)
        network_edges = gpd.read_parquet(edges_path)

        return network_nodes, network_edges

    overture = Overture.load(data.locations_gdf, cfg.inputs.overture, root)

    subsector_data = data.with_filter("subsector")
    network_nodes, network_edges = build_network_graph(subsector_data.locations_gdf, overture, cfg.inputs.statistics)

    network_path.mkdir(parents=True, exist_ok=True)
    network_nodes.to_parquet(nodes_path)
    network_edges.to_parquet(edges_path)

    return network_nodes, network_edges


def load_gva_network_graph(
    gva_data: GenevaData,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Wrapper around ``load_network_graph`` for the Geneva dataset."""
    return load_network_graph(gva_data, cfg, build_gva_network_graph, project_root, name)


def load_toronto_network_graph(
    toronto_data: TorontoData,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Wrapper around ``load_network_graph`` for the Toronto dataset."""
    return load_network_graph(toronto_data, cfg, build_toronto_network_graph, project_root, name)


def build_gva_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build the Geneva spatial contiguity graph enriched with census stats, POIs, and land uses.

    Disconnected island subsectors are connected to the mainland via a kNN graph.
    """
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    stats_cfg: GenevaStatsInputs

    network_locations = add_geneva_population_job_statistics(locations, stats_cfg)
    network_locations = overture.add_poi_counts(network_locations)
    network_locations = overture.add_land_uses(network_locations)
    network_locations = network_locations.set_index("loc_id")

    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    # Connect disconnected subsectors to main graph
    island_loc_ids = [
        "subsector-174",
        "subsector-141",
        "subsector-10",
        "subsector-325",
    ]
    cross_lake_loc_ids = ["subsector-243", "subsector-261", "subsector-54"]
    island_nodes = nodes[nodes.index.isin(island_loc_ids)]
    mainland_nodes = nodes[~nodes.index.isin(island_loc_ids) & ~nodes.index.isin(cross_lake_loc_ids)]

    _, island_edges = c2g.knn_graph(island_nodes, k=3, target_gdf=mainland_nodes)
    island_edges = island_edges.reset_index()
    island_edges["source"] = island_edges["source"].str[1]
    island_edges["target"] = island_edges["target"].str[1]
    island_edges = island_edges.set_index(["source", "target"])

    nodes = nodes.to_crs(CRS)
    edges: gpd.GeoDataFrame = pd.concat([edges, island_edges])
    edges = edges.to_crs(CRS)

    return nodes, edges


def build_toronto_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build the Toronto spatial contiguity graph enriched with census stats, POIs, and land uses."""
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    stats_cfg: TorontoStatsInputs

    network_locations = add_toronto_population_job_statistics(locations, stats_cfg)
    network_locations = overture.add_poi_counts(network_locations)
    network_locations = overture.add_land_uses(network_locations)
    network_locations = network_locations.set_index("loc_id")

    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    nodes = nodes.to_crs(CRS)
    edges: gpd.GeoDataFrame = pd.concat([edges])
    edges = edges.to_crs(CRS)

    return nodes, edges


# =========================================
# B. Spatial demographics
# =========================================


def add_user_cols(
    user_id: str,
    network_nodes: gpd.GeoDataFrame,
    location_visits: pl.DataFrame,
    home_locations: pl.DataFrame,
    work_locations: pl.DataFrame,
    edu_locations: pl.DataFrame,
) -> gpd.GeoDataFrame:
    """Annotate network nodes with per-user binary indicators and visit purposes.

    Adds columns ``is_home``, ``is_work``, ``is_edu``, ``is_visited``, and ``purpose``
    (comma-joined string of visit purposes) to a copy of ``network_nodes``.
    """
    network_nodes = network_nodes.copy().reset_index()

    user_visits = location_visits.filter(user_id=user_id)["loc_id"].to_list()
    home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
    work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
    edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()

    network_nodes["is_home"] = network_nodes["loc_id"].isin(home_location).astype(int)
    network_nodes["is_work"] = network_nodes["loc_id"].isin(work_location).astype(int)
    network_nodes["is_edu"] = network_nodes["loc_id"].isin(edu_location).astype(int)
    network_nodes["is_visited"] = network_nodes["loc_id"].isin(user_visits).astype(int)

    visit_purposes = (
        location_visits
        .filter(user_id=user_id)
        .select("loc_id", pl.col("purpose"))
        .group_by("loc_id")
        .agg(pl.col("purpose").str.join(", "))
        .to_pandas()
    )
    network_nodes = network_nodes.merge(visit_purposes, on="loc_id", how="left")
    network_nodes = network_nodes.set_index("loc_id").sort_index()

    return network_nodes


def create_spatial_demographics(data: NetworkData) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-user spatial feature and label tensors over all network nodes.

    Returns:
        Tuple ``(spatial_features, spatial_labels)`` of shapes
        ``[n_users, n_nodes, n_features]`` and ``[n_users, n_nodes, 1]``.
        Features contain ``is_home``; labels contain ``is_visited``.
    """
    locations = data.locations_df.select("loc_id").with_row_index("loc_order")
    all_users_and_locs = data.user_ids.to_frame().join(locations, how="cross").lazy()

    n_users = len(data.user_ids)
    n_locs = len(locations)

    # Build spatial features (i.e. will be appended to PyG data.x)
    feature_df = (
        all_users_and_locs
        .pipe(add_indicator_column, data.home_locations, "is_home")
        .pipe(add_indicator_column, data.work_locations, "is_work")
        .pipe(add_indicator_column, data.edu_locations, "is_edu")
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
        .pipe(add_indicator_column, data.location_visits, "is_visited")
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    label_cols = ["is_visited"]
    spatial_labels = torch.tensor(label_df.select(label_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(label_cols)
    )

    return spatial_features, spatial_labels


def create_home_distances(data, network_nodes: gpd.GeoDataFrame) -> torch.Tensor:
    """Build a distance-from-home tensor over all network nodes for every user. Uses Euclidian distance between
    centroids.

    Returns:
        Float tensor of shape ``[n_users, n_nodes, 1]``.
    """
    nodes = network_nodes.sort_index().to_crs(network_nodes.estimate_utm_crs())
    coords = nodes.geometry.get_coordinates()

    home_by_user = dict(data.home_locations.iter_rows())

    n_users, n_nodes = len(data.user_ids), len(coords)
    distances = np.zeros((n_users, n_nodes, 1), dtype=np.float32)

    for user_idx, user_id in enumerate(data.user_ids):
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
# C. Individual demographics
# =========================================


def create_individual_demographics(data: NetworkData) -> torch.Tensor:
    """Create the demographic tensor from NetworkData, returns a float32 tensor of shape ``[n_users, n_demo_features]``.
    If there are no demographics (e.g. Geneva), returns a (n_users, 1) dummy tensor of ones."""
    indiv_demographics = data.users_df.drop("user_id", "home_loc_id")

    if len(indiv_demographics) == 0:
        return torch.ones((len(data.user_ids), 1), dtype=torch.float32)

    return torch.tensor(indiv_demographics.to_numpy(), dtype=torch.float32)


# =========================================
# PyG / Torch creation
# =========================================


def convert_to_torch(
    data: NetworkData, network_nodes: gpd.GeoDataFrame, network_edges: gpd.GeoDataFrame
) -> tuple[pyg.data.Data | pyg.data.HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert network graph and network data into PyG and tensor form.

    Returns:
        Tuple ``(network_graph, spatial_features, spatial_labels, demographics, distances)``:
        the shared PyG graph, per-user spatial features ``[n_users, n_nodes, n_feat]``,
        per-user labels ``[n_users, n_nodes, 1]``, individual demographics ``[n_users, n_demo]``,
        and per-user distance-from-home ``[n_users, n_nodes, 1]``.
    """
    spatial_features, spatial_labels = create_spatial_demographics(data)
    demographics = create_individual_demographics(data)
    distances = create_home_distances(data, network_nodes)

    node_feature_cols = [col for col in network_nodes.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    network_graph = c2g.gdf_to_pyg(
        network_nodes,
        network_edges,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=["weight"],
        keep_geom=False,
        device="cpu",
    )

    return network_graph, spatial_features, spatial_labels, demographics, distances


def load_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "Graphs",
):
    """Load a list of per-user PyG graphs from pickle cache, building and caching on first call."""
    dataset_path = get_project_root(project_root) / cfg.paths.pyg_datasets / f"{name}.pickle"

    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    graphs = convert_to_pyg_graphs(network_data, network_nodes, network_edges)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as f:
        pickle.dump(graphs, f)

    return graphs


def convert_to_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
):
    """Build one annotated PyG graph per user in parallel using ``joblib``."""
    visits = network_data.location_visits
    home_locations = network_data.home_locations
    work_locations = network_data.work_locations
    edu_locations = network_data.edu_locations

    def build_graph(user_id: str):
        return build_user_pyg_graph(
            user_id,
            network_nodes,
            network_edges,
            visits,
            home_locations,
            work_locations,
            edu_locations,
        )

    print("Starting graph generation... Total: ", len(network_data.user_ids))

    parallel = Parallel(n_jobs=-1, verbose=1)
    graphs = parallel(delayed(build_graph)(user_id) for user_id in network_data.user_ids)

    print("Graph generation finished.")

    return graphs


def build_user_pyg_graph(
    user_id: str,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    location_visits: pl.DataFrame,
    home_locations: pl.DataFrame,
    work_locations: pl.DataFrame,
    edu_locations: pl.DataFrame,
):
    """Build a single user's PyG graph with user-specific indicators and positional encodings.

    Applies random-walk PE (length 20) and Laplacian eigenvector PE (k=8) transforms.
    """
    indiv_nodes = add_user_cols(
        user_id,
        network_nodes,
        location_visits,
        home_locations,
        work_locations,
        edu_locations,
    )
    node_feature_cols = [col for col in indiv_nodes.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    indiv_graph = c2g.gdf_to_pyg(
        indiv_nodes,
        network_edges,
        node_feature_cols=node_feature_cols,
        node_label_cols=["is_visited"],
        edge_feature_cols=["weight"],
        keep_geom=False,
        device="cpu",
    )

    indiv_graph.user_id = user_id

    if indiv_graph.num_nodes != len(network_nodes):
        raise ValueError(f"Invalid graph number of nodes {len(indiv_nodes)} for {user_id=}.")

    transforms = T.Compose([
        T.AddRandomWalkPE(walk_length=20, attr_name=None),
        T.AddLaplacianEigenvectorPE(k=8, attr_name=None),
    ])

    return transforms(indiv_graph)


def load_data(cfg: DataConfig, project_root: Path | None = None):
    if cfg.name == "THATS":
        cfg = cast(TorontoDataConfig, cfg)
        return _load_toronto_data(cfg, project_root)
    elif cfg.name == "GenevaTPG":
        cfg = cast(TorontoDataConfig, cfg)
        return _load_geneva_data(cfg, project_root)

    raise ValueError(f"Unknown Dataset {cfg.name}")


def _load_toronto_data(cfg: TorontoDataConfig, project_root: Path | None = None):
    data = TorontoData.load(cfg, project_root).with_filter("subsector")
    network_nodes, network_edges = load_toronto_network_graph(data, cfg, project_root)

    return data, network_nodes, network_edges


def _load_geneva_data(cfg: GenevaDataConfig, project_root: Path | None = None):
    data = GenevaData.load(cfg, project_root).with_filter("subsector")
    network_nodes, network_edges = load_gva_network_graph(data, cfg, project_root)

    return data, network_nodes, network_edges
