"""Tests of the rail edges on a small feed over three zones in a row."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import pytest
import torch
from shapely.geometry import box

from activitygraphs.base import CRS
from activitygraphs.config import GTFSFeeds
from activitygraphs.graph.rail import EDGE_INDEX_NAMES, RAIL_EDGE_SCHEMA, build_rail_edges
from activitygraphs.surveys.gtfs import GTFSTables, read_gtfs
from activitygraphs.tensors.build import _build_edges

SERVICE_DATE = "2022-10-19"
UTM_CRS = "EPSG:32617"
ZONE_IDS = ["west", "mid", "east"]

FEED = {
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
    "A,A,43.645,-79.395,1,\n"
    "A_P1,A platform 1,43.645,-79.385,0,A\n"
    "B,B,43.645,-79.385,0,\n"
    "C,C,43.645,-79.375,0,\n"
    "C2,C2,43.645,-79.372,0,\n"
    "D,D,43.645,-79.300,0,\n",
    "routes.txt": "route_id,route_short_name,route_long_name,route_type\n"
    "LINE1,1,Line 1,1\n"
    "LINE2,2,Line 2,1\n"
    "GO,LW,Lakeshore West,2\n"
    "BUS,504,King,3\n",
    "trips.txt": "route_id,service_id,trip_id\n"
    "LINE1,WEEKDAY,line1-out\n"
    "LINE1,WEEKDAY,line1-back\n"
    "GO,WEEKDAY,go-express\n"
    "GO,SUNDAY,go-sunday\n"
    "BUS,WEEKDAY,bus\n",
    "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "line1-out,07:00:00,07:00:00,A_P1,1\n"
    "line1-out,07:05:00,07:05:00,B,2\n"
    "line1-out,07:12:00,07:12:00,C,3\n"
    "line1-out,07:20:00,07:20:00,D,4\n"
    "line1-back,08:00:00,08:00:00,C,1\n"
    "line1-back,08:07:00,08:07:00,B,2\n"
    "line1-back,08:12:00,08:12:00,A_P1,3\n"
    "go-express,07:00:00,07:00:00,A_P1,1\n"
    "go-express,07:08:00,07:08:00,C,2\n"
    "go-express,07:10:00,07:10:00,C2,3\n"
    "go-sunday,07:00:00,07:00:00,A_P1,1\n"
    "go-sunday,07:01:00,07:01:00,B,2\n"
    "bus,07:00:00,07:00:00,A_P1,1\n"
    "bus,07:01:00,07:01:00,B,2\n",
    "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WEEKDAY,1,1,1,1,1,0,0,20220101,20221231\n"
    "SUNDAY,0,0,0,0,0,0,1,20220101,20221231\n",
}

UNTIMED_TRIP = {
    "trips.txt": "LINE2,WEEKDAY,line2\n",
    "stop_times.txt": "line2,10:00:00,10:00:00,A_P1,1\nline2,,,B,2\nline2,10:04:00,10:04:00,C,3\n",
}


def read_feed(tmp_path: Path, extra_rows: dict[str, str]) -> GTFSTables:
    feed_dir = tmp_path / "feed"
    feed_dir.mkdir()
    for name, text in FEED.items():
        (feed_dir / name).write_text(text + extra_rows.get(name, ""))

    return read_gtfs(GTFSFeeds(service_date=SERVICE_DATE, feeds={"ttc": "feed"}), tmp_path)


def edges_by_pair(edges: pd.DataFrame) -> dict[tuple[str, str], tuple[float, int]]:
    rows = zip(edges.index, edges["rail_time_s"], edges["rail_trips"], strict=True)
    return {pair: (time_s, trips) for pair, time_s, trips in rows}


@pytest.fixture
def zones() -> gpd.GeoDataFrame:
    """Three zones of 0.01 degrees in a row, with stop D outside every zone.

    ``west`` holds station A, ``mid`` holds B and the platform of A, and ``east`` holds C and C2.
    """
    zone_boxes = [box(-79.40 + 0.01 * i, 43.64, -79.39 + 0.01 * i, 43.65) for i in range(len(ZONE_IDS))]
    polygons = gpd.GeoSeries(zone_boxes, index=pd.Index(ZONE_IDS, name="loc_id"), crs=CRS)
    projected_polygons = polygons.to_crs(UTM_CRS)

    return gpd.GeoDataFrame(
        {"original_geometry": projected_polygons}, geometry=projected_polygons.centroid, crs=UTM_CRS
    )


@pytest.fixture
def edges(tmp_path, zones) -> dict[tuple[str, str], tuple[float, int]]:
    return edges_by_pair(build_rail_edges(zones, read_feed(tmp_path, {})))


class TestBuildRailEdges:
    def test_join_the_zones_of_every_pair_of_stations_on_a_rail_trip(self, edges):
        assert set(edges) == {("west", "mid"), ("west", "east"), ("mid", "east")}

    def test_take_the_median_time_of_a_route_and_the_shortest_route(self, edges):
        times = {pair: time_s for pair, (time_s, _) in edges.items()}

        assert times == {("west", "mid"): 300.0, ("west", "east"): 480.0, ("mid", "east"): 420.0}

    def test_count_the_rail_trips_of_every_route_on_the_service_date(self, edges):
        trips = {pair: n_trips for pair, (_, n_trips) in edges.items()}

        assert trips == {("west", "mid"): 2, ("west", "east"): 3, ("mid", "east"): 2}

    def test_an_untimed_call_takes_a_time_interpolated_along_its_trip(self, tmp_path, zones):
        edges = edges_by_pair(build_rail_edges(zones, read_feed(tmp_path, UNTIMED_TRIP)))

        assert edges[("west", "mid")] == (120.0, 3)
        assert edges[("mid", "east")] == (120.0, 3)
        assert edges[("west", "east")] == (240.0, 4)

    def test_zones_without_a_crs_raise(self, tmp_path, zones):
        with pytest.raises(ValueError):
            build_rail_edges(zones.set_crs(None, allow_override=True), read_feed(tmp_path, {}))


class TestRailEdgeTensors:
    def test_hold_each_edge_in_both_directions_with_its_columns(self):
        pair_index = pd.MultiIndex.from_tuples([("west", "east"), ("mid", "east")], names=EDGE_INDEX_NAMES)
        rail_edges = pd.DataFrame({"rail_time_s": [480.0, 420.0], "rail_trips": [3, 2]}, index=pair_index)
        node_ids = pl.DataFrame({"loc_id": ZONE_IDS})

        edges = _build_edges(rail_edges, node_ids, RAIL_EDGE_SCHEMA)
        edge_index, edge_attr = edges.index, edges.attr

        assert edge_index.tolist() == [[0, 1, 2, 2], [2, 2, 0, 1]]
        assert edge_attr.shape == (4, RAIL_EDGE_SCHEMA.width)
        torch.testing.assert_close(edge_attr[:2], edge_attr[2:])
        torch.testing.assert_close(edge_attr[0], torch.tensor([480.0, 3.0]))
