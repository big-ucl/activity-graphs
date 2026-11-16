"""Tests of the GTFS feed subsets routed by R5, and of the travel-time table built from an R5 matrix."""

import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Point, box

from activitygraphs.base import CRS
from activitygraphs.config import GTFSFeeds
from activitygraphs.graph.rail import EDGE_INDEX_NAMES
from activitygraphs.graph.travel_times import (
    MAX_TRAVEL_TIME,
    _is_walkable,
    _nearest_walkable_points,
    _pt_time_table,
    _replace_pairs,
    _unlinked_zone_ids,
)
from activitygraphs.surveys.gtfs import write_feed_subsets

SERVICE_DATE = "2022-05-18"
CLIP_POLYGON = box(6.0, 46.0, 6.2, 46.2)

FEED = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\nTPG,TPG,https://tpg.ch,Europe/Zurich\n",
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
    "PA,Station A,46.30,6.30,1,\n"
    "A,A,46.10,6.10,0,PA\n"
    "B,B,46.15,6.15,0,\n"
    "C,C,46.50,6.50,0,\n"
    "D,D,46.60,6.60,0,\n",
    "routes.txt": "route_id,agency_id,route_short_name,route_type\nR1,TPG,1,3\nR2,TPG,2,3\n",
    "trips.txt": "route_id,service_id,trip_id,shape_id\n"
    "R1,WEEKDAY,in-and-out,S1\n"
    "R2,WEEKDAY,outside,S2\n"
    "R1,SUNDAY,sunday,S1\n",
    "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "in-and-out,07:00:00,07:00:00,A,1\n"
    "in-and-out,07:30:00,07:30:00,C,2\n"
    "outside,07:00:00,07:00:00,C,1\n"
    "outside,07:10:00,07:10:00,D,2\n"
    "sunday,07:00:00,07:00:00,A,1\n"
    "sunday,07:05:00,07:05:00,B,2\n",
    "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WEEKDAY,1,1,1,1,1,0,0,20220101,20221231\n"
    "SUNDAY,0,0,0,0,0,0,1,20220101,20221231\n",
    "calendar_dates.txt": "service_id,date,exception_type\nWEEKDAY,20220518,1\nWEEKDAY,20221225,2\nSUNDAY,20220518,2\n",
    "shapes.txt": "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\nS1,46.1,6.1,1\nS2,46.5,6.5,1\n",
    "transfers.txt": "from_stop_id,to_stop_id,transfer_type\nA,C,0\nC,D,0\n",
}


OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" version="1" lat="46.2001" lon="6.0900"/>
  <node id="2" version="1" lat="46.2001" lon="6.1100"/>
  <node id="3" version="1" lat="46.2010" lon="6.0900"/>
  <node id="4" version="1" lat="46.2010" lon="6.1100"/>
  <way id="10" version="1"><nd ref="1"/><nd ref="2"/><tag k="highway" v="motorway"/></way>
  <way id="11" version="1"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
</osm>
"""


def write_feed(tmp_path, replaced_tables: dict[str, bytes]) -> GTFSFeeds:
    feed_dir = tmp_path / "feed"
    feed_dir.mkdir()
    for name, text in FEED.items():
        (feed_dir / name).write_bytes(replaced_tables.get(name, text.encode()))

    return GTFSFeeds(service_date=SERVICE_DATE, feeds={"tpg": "feed"})


def write_subset(tmp_path, replaced_tables: dict[str, bytes] | None = None) -> dict[str, pl.DataFrame]:
    gtfs_feeds = write_feed(tmp_path, replaced_tables or {})
    (feed_path,) = write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, tmp_path / "subsets")

    with zipfile.ZipFile(feed_path) as archive:
        return {
            name.removesuffix(".txt"): pl.read_csv(io.BytesIO(archive.read(name)), infer_schema=False)
            for name in archive.namelist()
        }


class TestWriteFeedSubsets:
    def test_keep_the_trips_of_the_service_date_that_call_inside_the_polygon(self, tmp_path):
        tables = write_subset(tmp_path)

        assert tables["trips"]["trip_id"].to_list() == ["in-and-out"]
        assert tables["routes"]["route_id"].to_list() == ["R1"]

    def test_keep_every_call_of_a_kept_trip_and_the_parent_stations(self, tmp_path):
        tables = write_subset(tmp_path)

        assert tables["stop_times"]["stop_id"].to_list() == ["A", "C"]
        assert sorted(tables["stops"]["stop_id"].to_list()) == ["A", "C", "PA"]

    def test_keep_the_rows_of_the_other_tables_whose_ids_are_kept(self, tmp_path):
        tables = write_subset(tmp_path)

        assert tables["shapes"]["shape_id"].to_list() == ["S1"]
        assert tables["transfers"]["to_stop_id"].to_list() == ["C"]
        assert tables["calendar"]["service_id"].to_list() == ["WEEKDAY"]
        assert tables["agency"]["agency_id"].to_list() == ["TPG"]

    def test_keep_the_calendar_dates_of_the_service_date_only(self, tmp_path):
        tables = write_subset(tmp_path)

        assert tables["calendar_dates"]["date"].to_list() == ["20220518"]

    def test_a_table_left_with_no_row_is_not_written(self, tmp_path):
        calendar_dates = b"service_id,date,exception_type\nWEEKDAY,20221225,2\n"

        tables = write_subset(tmp_path, {"calendar_dates.txt": calendar_dates})

        assert "calendar_dates" not in tables
        assert tables["trips"]["trip_id"].to_list() == ["in-and-out"]

    def test_a_feed_without_a_trip_inside_the_polygon_is_not_written(self, tmp_path):
        gtfs_feeds = write_feed(tmp_path, {})
        far_polygon = box(10.0, 50.0, 10.1, 50.1)

        assert write_feed_subsets(gtfs_feeds, tmp_path, far_polygon, tmp_path / "subsets") == []

    def test_fields_beyond_the_header_are_dropped(self, tmp_path):
        agency = FEED["agency.txt"].replace("Europe/Zurich\n", "Europe/Zurich,,\n").encode()

        tables = write_subset(tmp_path, {"agency.txt": agency})

        assert tables["agency"].columns == ["agency_id", "agency_name", "agency_url", "agency_timezone"]

    def test_bytes_that_are_not_utf8_are_replaced(self, tmp_path):
        agency = FEED["agency.txt"].replace("TPG,TPG,", "TPG,Gen\xe8ve,").encode("latin-1")

        tables = write_subset(tmp_path, {"agency.txt": agency})

        assert tables["agency"]["agency_name"].to_list() == ["Gen\ufffdve"]


class TestPTTimeTable:
    def test_an_unreached_pair_takes_the_cap_and_is_flagged(self):
        matrix = pd.DataFrame({"from_id": ["a", "a", "b"], "to_id": ["a", "b", "a"], "travel_time": [0, 25, np.nan]})

        times = _pt_time_table(matrix)

        assert times.index.names == ["source", "target"]
        assert times["pt_time_min"].tolist() == [0.0, 25.0, MAX_TRAVEL_TIME.total_seconds() / 60]
        assert times["pt_reachable"].tolist() == [True, True, False]


def pair_table(rows: list[tuple[str, str, float, bool]]) -> pd.DataFrame:
    pair_index = pd.MultiIndex.from_tuples([(source, target) for source, target, _, _ in rows], names=EDGE_INDEX_NAMES)
    pair_columns = {"pt_time_min": [row[2] for row in rows], "pt_reachable": [row[3] for row in rows]}

    return pd.DataFrame(pair_columns, index=pair_index)


class TestUnlinkedZones:
    def test_a_zone_that_reaches_no_other_zone_is_unlinked(self):
        times = pair_table([
            ("a", "a", 0.0, True),
            ("a", "b", 20.0, True),
            ("b", "a", 180.0, False),
            ("b", "b", 0.0, True),
        ])

        assert _unlinked_zone_ids(times) == ["b"]


class TestReplacePairs:
    def test_rerouted_pairs_replace_the_first_routing_in_place(self):
        times = pair_table([("a", "a", 0.0, True), ("a", "b", 180.0, False), ("b", "a", 180.0, False)])
        from_b = pair_table([("b", "a", 30.0, True)])
        to_b = pair_table([("a", "b", 25.0, True)])

        replaced = _replace_pairs(times, [from_b, to_b])

        assert replaced.index.equals(times.index)
        assert replaced["pt_time_min"].tolist() == [0.0, 25.0, 30.0]
        assert replaced["pt_reachable"].tolist() == [True, True, True]


class TestNearestWalkablePoints:
    def test_a_point_beside_a_motorway_moves_to_the_nearest_walkable_way(self, tmp_path):
        osm_path = tmp_path / "streets.osm"
        osm_path.write_text(OSM_XML)
        points = gpd.GeoSeries([Point(6.1000, 46.2000)], index=["zone"], crs=CRS)

        moved, distance_m = _nearest_walkable_points(points, osm_path)

        assert moved.index.tolist() == ["zone"]
        assert moved.iloc[0].y == pytest.approx(46.2010, abs=1e-6)
        assert moved.iloc[0].x == pytest.approx(6.1000, abs=1e-6)
        assert distance_m[0] == pytest.approx(111, abs=2)

    @pytest.mark.parametrize(
        ("tags", "walkable"),
        [
            ({"highway": "residential"}, True),
            ({"highway": "motorway"}, False),
            ({"highway": "trunk_link"}, False),
            ({"highway": "footway", "foot": "no"}, False),
            ({"highway": "service", "access": "private"}, False),
            ({"highway": "service", "access": "private", "foot": "yes"}, True),
        ],
    )
    def test_motorways_and_ways_closed_to_walking_are_not_walkable(self, tags, walkable):
        assert _is_walkable(tags) == walkable
