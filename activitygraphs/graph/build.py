"""Load and build the contiguity graph over the zones of a Survey, with node features, rail edges and PT travel times."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

import city2graph as c2g
import geopandas as gpd
import pandas as pd

from activitygraphs.base import CRS
from activitygraphs.config import DataConfig, PTPaths
from activitygraphs.graph.access import add_access_features, read_motorway_junctions
from activitygraphs.graph.rail import EDGE_INDEX_NAMES, build_rail_edges
from activitygraphs.graph.travel_times import build_pt_travel_times
from activitygraphs.surveys.base import SurveyData, SurveyLoader
from activitygraphs.surveys.gtfs import GTFSTables, read_gtfs, write_feed_subsets
from activitygraphs.surveys.overture import Overture
from activitygraphs.surveys.registry import survey_loader
from activitygraphs.utils import get_project_root

GRAPH_DIR = "Graph"
PT_FEEDS_DIR = "pt_feeds"


class GraphFile(StrEnum):
    """Filenames of a ``GraphTables`` bundle, in the cache directory."""

    NODES = "nodes.parquet"
    CONTIGUITY_EDGES = "contiguity_edges.parquet"
    RAIL_EDGES = "rail_edges.parquet"
    PT_TRAVEL_TIMES = "pt_travel_times.parquet"


@dataclass(frozen=True)
class GraphTables:
    """The tables of the location graph for a survey.

    Edges tables are indexed by (source, target) ``loc_id``, are undirected and contain additional edge features.

    Attributes:
        nodes: Zones indexed by ``loc_id``, with their features.
        contiguity_edges: Contiguity edges table.
        rail_edges: Rail edges table.
        pt_travel_times: Public transport travel time between every ordered pair of zones, indexed by (source,
            target) ``loc_id``.
    """

    nodes: gpd.GeoDataFrame
    contiguity_edges: gpd.GeoDataFrame
    rail_edges: pd.DataFrame
    pt_travel_times: pd.DataFrame

    def save(self, directory: Path) -> None:
        """Write the tables to ``directory``, one file each."""
        directory.mkdir(parents=True, exist_ok=True)
        self.nodes.to_parquet(directory / GraphFile.NODES)
        self.contiguity_edges.to_parquet(directory / GraphFile.CONTIGUITY_EDGES)
        self.rail_edges.to_parquet(directory / GraphFile.RAIL_EDGES)
        self.pt_travel_times.to_parquet(directory / GraphFile.PT_TRAVEL_TIMES)

    @classmethod
    def read(cls, directory: Path) -> Self:
        """Read the tables written by ``save`` to ``directory``."""
        return cls(
            nodes=gpd.read_parquet(directory / GraphFile.NODES),
            contiguity_edges=gpd.read_parquet(directory / GraphFile.CONTIGUITY_EDGES),
            rail_edges=pd.read_parquet(directory / GraphFile.RAIL_EDGES),
            pt_travel_times=pd.read_parquet(directory / GraphFile.PT_TRAVEL_TIMES),
        )


def load_graph(survey: SurveyData, cfg: DataConfig, project_root: Path | None = None) -> GraphTables:
    """Load the graph from cache, or build and cache it.

    Cached parquet files are stored under ``cfg.paths.processed/Graph/``.

    Args:
        survey: Survey from which to build the graph, nodes are survey ``zone``.
        cfg: Dataset config providing the survey key and the processed and raw paths.
        project_root: Repo root, defaults to ``Path(".")``.

    Returns:
        The tables representing the graph (nodes, edges and public transport travel times).
    """
    root = get_project_root(project_root)
    graph_path = root / cfg.paths.processed / GRAPH_DIR

    if not all((graph_path / file).exists() for file in GraphFile):
        loader = survey_loader(cfg)
        zones_gdf = survey.with_filter("zone").locations_gdf
        overture = Overture.load(zones_gdf, cfg.inputs.overture, root)
        statistics = loader.read_statistics(cfg, root)

        pt_inputs = cast(PTPaths, cfg.inputs)
        gtfs = read_gtfs(pt_inputs.gtfs, root / cfg.paths.gtfs)

        osm_root_dir = root / cfg.paths.osm
        osm_path = osm_root_dir / pt_inputs.osm_extract

        junctions = read_motorway_junctions(osm_path)
        clip_polygon = gpd.read_file(osm_root_dir / pt_inputs.osm_clip_polygon).union_all()

        pt_feeds_dir = graph_path / PT_FEEDS_DIR
        pt_feed_paths = write_feed_subsets(pt_inputs.gtfs, root / cfg.paths.gtfs, clip_polygon, pt_feeds_dir)

        graph = _build_graph(zones_gdf, overture, statistics, loader, gtfs, junctions, osm_path, pt_feed_paths)
        graph.save(graph_path)

    return GraphTables.read(graph_path)


def _build_graph[S](
    zones_gdf: gpd.GeoDataFrame,
    overture: Overture,
    statistics: S,
    loader: SurveyLoader[Any, S],
    gtfs: GTFSTables,
    junctions: gpd.GeoSeries,
    osm_path: Path,
    pt_feed_paths: list[Path],
) -> GraphTables:
    """Build the spatial contiguity graph over ``zones_gdf``, enriched with census stats, POIs, land uses and access.

    Args:
        zones_gdf: Zone locations, with polygon geometry and a ``loc_id`` column.
        overture: Overture POIs and land uses covering the zones.
        statistics: Census statistics from ``loader.read_statistics``.
        loader: The survey's loader, which adds the statistics and any extra edges.
        gtfs: The GTFS feeds of the dataset, read for the rail stations, the peak departures and the rail edges.
        junctions: The motorway junctions around the zones.
        osm_path: The OSM extract of the street network, read for the public transport travel times.
        pt_feed_paths: The zipped GTFS feeds routed for the public transport travel times.

    Returns:
        The nodes, the contiguity edges, the rail edges and the public transport travel times of the graph, the nodes
        and the contiguity edges in ``CRS``.
    """
    utm_crs = zones_gdf.estimate_utm_crs()
    zones_gdf = zones_gdf.to_crs(utm_crs)

    zones_gdf = loader.add_statistics(zones_gdf, statistics)
    zones_gdf = overture.add_poi_counts(zones_gdf)
    zones_gdf = overture.add_land_uses(zones_gdf)
    zones_gdf = zones_gdf.set_index("loc_id")

    nodes_gdf, contiguity_edges_gdf = c2g.contiguity_graph(zones_gdf, set_point_nodes=True)

    if loader.build_extra_edges is not None:
        extra_edges = loader.build_extra_edges(nodes_gdf, contiguity_edges_gdf)
        contiguity_edges_gdf = cast(gpd.GeoDataFrame, pd.concat([contiguity_edges_gdf, extra_edges]))
        contiguity_edges_gdf = c2g.canonicalize_edges(contiguity_edges_gdf)

    nodes_gdf = add_access_features(nodes_gdf, gtfs, junctions)
    rail_edges = build_rail_edges(nodes_gdf, gtfs)
    pt_travel_times = build_pt_travel_times(nodes_gdf, osm_path, pt_feed_paths, gtfs.service_date)

    nodes = nodes_gdf.to_crs(CRS)
    contiguity_edges = contiguity_edges_gdf.to_crs(CRS).rename_axis(EDGE_INDEX_NAMES)

    return GraphTables(
        nodes=nodes, contiguity_edges=contiguity_edges, rail_edges=rail_edges, pt_travel_times=pt_travel_times
    )
