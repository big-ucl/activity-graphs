"""Load and build the contiguity graph over the zones of a Survey, with enriched node features."""

from pathlib import Path
from typing import Any, cast

import city2graph as c2g
import geopandas as gpd
import pandas as pd

from activitygraphs.base import CRS
from activitygraphs.config import DataConfig
from activitygraphs.surveys.base import SurveyData, SurveyLoader
from activitygraphs.surveys.overture import Overture
from activitygraphs.surveys.registry import survey_loader
from activitygraphs.utils import get_project_root

GRAPH_DIR = "Graph"


def load_graph(
    survey: SurveyData, cfg: DataConfig, project_root: Path | None = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the graph from cache, or build and cache it.

    Cached parquet files are stored under ``cfg.paths.processed/Graph/``.

    Args:
        survey: Survey from which to build the graph, nodes are survey ``zone``.
        cfg: Dataset config providing the survey key and the processed and raw paths.
        project_root: Repo root, defaults to ``Path(".")``.

    Returns:
        Tuple ``(nodes_gdf, edges_gdf)``: nodes indexed by ``loc_id``, edges by (source, target) ``loc_id``.
    """
    root = get_project_root(project_root)
    graph_path = root / cfg.paths.processed / GRAPH_DIR
    nodes_path = graph_path / "nodes.parquet"
    edges_path = graph_path / "edges.parquet"

    if nodes_path.exists() and edges_path.exists():
        return gpd.read_parquet(nodes_path), gpd.read_parquet(edges_path)

    loader = survey_loader(cfg)
    zones_gdf = survey.with_filter("zone").locations_gdf
    overture = Overture.load(zones_gdf, cfg.inputs.overture, root)
    statistics = loader.read_statistics(cfg, root)

    nodes_gdf, edges_gdf = build_graph(zones_gdf, overture, statistics, loader)

    graph_path.mkdir(parents=True, exist_ok=True)
    nodes_gdf.to_parquet(nodes_path)
    edges_gdf.to_parquet(edges_path)

    return nodes_gdf, edges_gdf


def build_graph[S](
    zones_gdf: gpd.GeoDataFrame,
    overture: Overture,
    statistics: S,
    loader: SurveyLoader[Any, S],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build the spatial contiguity graph over ``zones_gdf``, enriched with census stats, POIs, and land uses.

    Args:
        zones_gdf: Zone locations, with polygon geometry and a ``loc_id`` column.
        overture: Overture POIs and land uses covering the zones.
        statistics: Census statistics from ``loader.read_statistics``.
        loader: The survey's loader, which adds the statistics and any extra edges.

    Returns:
        Tuple ``(nodes_gdf, edges_gdf)``: nodes indexed by ``loc_id``, edges by (source, target) ``loc_id``.
    """
    utm_crs = zones_gdf.estimate_utm_crs()
    zones_gdf = zones_gdf.to_crs(utm_crs)

    zones_gdf = loader.add_statistics(zones_gdf, statistics)
    zones_gdf = overture.add_poi_counts(zones_gdf)
    zones_gdf = overture.add_land_uses(zones_gdf)
    zones_gdf = zones_gdf.set_index("loc_id")

    nodes_gdf, edges_gdf = c2g.contiguity_graph(zones_gdf, set_point_nodes=True)

    if loader.build_extra_edges is not None:
        edges_gdf = cast(gpd.GeoDataFrame, pd.concat([edges_gdf, loader.build_extra_edges(nodes_gdf)]))
        edges_gdf = c2g.canonicalize_edges(edges_gdf)

    return nodes_gdf.to_crs(CRS), edges_gdf.to_crs(CRS)
