"""Tests of the GTFS feed subsets routed by R5, and of the travel-time table built from an R5 matrix."""

import io
import zipfile
from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Point, box

from activitygraphs.base import CRS
from activitygraphs.config import GTFSFeeds
from activitygraphs.graph import osrm
from activitygraphs.graph.osrm import _coord_strings, _duration_table, _foot_profile, _unlinked_rows
from activitygraphs.graph.rail import EDGE_INDEX_NAMES
from activitygraphs.graph.travel_times import (
    MAX_TRAVEL_TIME,
    WALK_SPEED_KMH,
    _is_walkable,
    _nearest_walkable_points,
    _pt_time_table,
    _replace_pairs,
    _unlinked_zone_ids,
    walkable_candidate_points,
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

TWO_STREETS_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" version="1" lat="46.2005" lon="6.0900"/>
  <node id="2" version="1" lat="46.2005" lon="6.1100"/>
  <node id="3" version="1" lat="46.2020" lon="6.0900"/>
  <node id="4" version="1" lat="46.2020" lon="6.1100"/>
  <way id="10" version="1"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
  <way id="11" version="1"><nd ref="1"/><nd ref="2"/><tag k="highway" v="service"/></way>
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

    def test_two_source_feeds_of_one_feed_name_get_different_file_names(self, tmp_path):
        gtfs_feeds = write_feed(tmp_path, {})
        (tmp_path / "other").mkdir()
        other_feeds = write_feed(tmp_path / "other", {})

        (feed_path,) = write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, tmp_path / "subsets")
        (other_path,) = write_feed_subsets(other_feeds, tmp_path / "other", CLIP_POLYGON, tmp_path / "other-subsets")

        assert feed_path.name.startswith("tpg-")
        assert other_path.name.startswith("tpg-")
        assert feed_path.name != other_path.name

    def test_writing_one_subset_again_gives_the_same_file_name(self, tmp_path):
        gtfs_feeds = write_feed(tmp_path, {})

        (first_path,) = write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, tmp_path / "first")
        (second_path,) = write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, tmp_path / "second")

        assert first_path.name == second_path.name

    def test_writing_the_subsets_again_leaves_one_zip_per_feed(self, tmp_path):
        gtfs_feeds = write_feed(tmp_path, {})
        out_dir = tmp_path / "subsets"
        write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, out_dir)
        (out_dir / "stale.zip").write_bytes(b"")

        (feed_path,) = write_feed_subsets(gtfs_feeds, tmp_path, CLIP_POLYGON, out_dir)

        assert list(out_dir.glob("*.zip")) == [feed_path]

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

    def test_the_candidate_points_lie_on_every_walkable_way_nearest_first(self, tmp_path):
        osm_path = tmp_path / "streets.osm"
        osm_path.write_text(TWO_STREETS_OSM_XML)
        points = gpd.GeoSeries([Point(6.1000, 46.2000)], index=["zone"], crs=CRS)

        candidates = walkable_candidate_points(points, osm_path, max_candidates=5)
        candidate_points, distance_m = candidates["zone"]

        assert candidate_points.y.tolist() == pytest.approx([46.2005, 46.2020], abs=1e-6)
        assert distance_m.tolist() == pytest.approx([56, 222], abs=2)

    def test_the_candidate_points_stop_at_the_largest_number_asked(self, tmp_path):
        osm_path = tmp_path / "streets.osm"
        osm_path.write_text(TWO_STREETS_OSM_XML)
        points = gpd.GeoSeries([Point(6.1000, 46.2000)], index=["zone"], crs=CRS)

        candidate_points, _ = walkable_candidate_points(points, osm_path, max_candidates=1)["zone"]

        assert candidate_points.y.tolist() == pytest.approx([46.2005], abs=1e-6)

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


class TestOSRMTables:
    def test_durations_become_minutes_capped_and_flagged(self):
        max_time_s = MAX_TRAVEL_TIME.total_seconds()
        durations_s = np.array([[0.0, 600.0], [np.nan, max_time_s + 60.0]])

        times = _duration_table(durations_s, np.array(["a", "b"]), "car")

        assert times.index.tolist() == [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
        assert times["car_time_min"].tolist() == [0.0, 10.0, 180.0, 180.0]
        assert times["car_reachable"].tolist() == [True, True, False, False]

    def test_the_foot_profile_walks_at_the_r5_walking_speed(self):
        profile = "function setup()\n  local walking_speed = 5\nend\n"

        assert f"local walking_speed = {WALK_SPEED_KMH}" in _foot_profile(profile)

    def test_a_foot_profile_without_the_speed_line_raises(self):
        with pytest.raises(ValueError):
            _foot_profile("function setup()\n  local walking_speed = 6\nend\n")

    def test_a_zone_that_reaches_no_other_zone_is_an_unlinked_row(self):
        durations_s = np.array([[0.0, 60.0, np.nan], [np.nan, 0.0, np.nan], [30.0, np.nan, 0.0]])

        assert _unlinked_rows(durations_s).tolist() == [1]


class TestRoadTravelTimeFallback:
    def test_every_zone_that_reaches_nothing_is_moved_to_its_first_linked_candidate(self, monkeypatch, tmp_path):
        zone_points = gpd.GeoSeries(
            [Point(6.10, 46.20), Point(6.11, 46.20), Point(6.12, 46.20), Point(6.13, 46.20)],
            index=["a", "b", "c", "d"],
            crs=CRS,
        )
        nodes_gdf = gpd.GeoDataFrame(geometry=zone_points)
        cut_off_candidates = gpd.GeoSeries([Point(6.12, 46.21), Point(6.13, 46.21)], crs=CRS)
        linked_candidates = gpd.GeoSeries([Point(6.12, 46.22), Point(6.13, 46.22)], crs=CRS)
        cut_off_coords = set(_coord_strings(zone_points.loc[["c", "d"]])) | set(_coord_strings(cut_off_candidates))

        def fake_route_table(base_url, coords, sources, destinations):
            destinations = np.arange(len(coords)) if destinations is None else destinations
            durations_s = np.full((len(sources), len(destinations)), 600.0)
            for i, source in enumerate(sources):
                for j, destination in enumerate(destinations):
                    if coords[source] in cut_off_coords or coords[destination] in cut_off_coords:
                        durations_s[i, j] = np.nan
            return durations_s

        def fake_candidates(points, osm_path, max_candidates):
            candidates = {}
            for position, label in enumerate(points.index):
                candidate_points = gpd.GeoSeries(
                    [cut_off_candidates.iloc[position], linked_candidates.iloc[position]], crs=CRS
                )
                candidates[label] = (candidate_points, np.array([10.0, 50.0]))
            return candidates

        @contextmanager
        def fake_server(mode_dir, num_zones):
            yield "http://osrm"

        monkeypatch.setattr(osrm, "_prepare_network", lambda mode, osm_path, mode_dir: None)
        monkeypatch.setattr(osrm, "_osrm_server", fake_server)
        monkeypatch.setattr(osrm, "_route_table", fake_route_table)
        monkeypatch.setattr(osrm, "walkable_candidate_points", fake_candidates)

        travel_times = osrm.build_road_travel_times(nodes_gdf, tmp_path / "extract.osm.pbf", tmp_path)

        for mode in osrm.OSRM_PROFILES:
            assert travel_times[f"{mode}_reachable"].all()
            assert travel_times.loc[("c", "a"), f"{mode}_time_min"] == 10.0
            assert travel_times.loc[("a", "d"), f"{mode}_time_min"] == 10.0
