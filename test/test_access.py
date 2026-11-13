"""Tests of the accessibility features on a small feed, two square zones and a small OSM extract."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from activitygraphs.base import CRS
from activitygraphs.config import GTFSFeeds
from activitygraphs.graph.access import (
    ACCESS_SCHEMA,
    add_access_features,
    peak_departures,
    rail_stations,
    read_motorway_junctions,
)
from activitygraphs.surveys.gtfs import GTFSTables, read_gtfs

SERVICE_DATE = "2022-10-19"
UTM_CRS = "EPSG:32617"

FEED = {
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
    "UNION,Union,43.6450,-79.3800,1,\n"
    "UNION_P1,Union platform 1,43.6451,-79.3801,0,UNION\n"
    "KING,King,43.6490,-79.3780,0,\n"
    "FAR,Far station,43.7500,-79.2000,0,\n"
    "BUS_A,Bus stop A,43.6455,-79.3790,0,\n"
    "BUS_B,Bus stop B,43.6460,-79.3700,0,\n"
    "BUS_C,Bus stop C,43.6465,-79.3600,0,\n",
    "routes.txt": "route_id,route_short_name,route_long_name,route_type\n"
    "LINE1,1,Line 1,1\n"
    "GO,LW,Lakeshore West,2\n"
    "BUS,504,King,3\n",
    "trips.txt": "route_id,service_id,trip_id\n"
    "LINE1,WEEKDAY,line1-am\n"
    "GO,SUNDAY,go-sunday\n"
    "BUS,WEEKDAY,bus-am\n"
    "BUS,WEEKDAY,bus-pm\n",
    "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "line1-am,07:10:00,07:10:00,UNION_P1,1\n"
    "line1-am,07:15:00,07:15:00,KING,2\n"
    "go-sunday,07:00:00,07:00:00,FAR,1\n"
    "go-sunday,07:30:00,07:30:00,UNION_P1,2\n"
    "bus-am,06:50:00,06:50:00,BUS_A,1\n"
    "bus-am,,,BUS_B,2\n"
    "bus-am,07:30:00,07:30:00,BUS_C,3\n"
    "bus-pm,17:00:00,17:00:00,BUS_A,1\n"
    "bus-pm,17:10:00,17:10:00,BUS_B,2\n",
    "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WEEKDAY,1,1,1,1,1,0,0,20220101,20221231\n"
    "SUNDAY,0,0,0,0,0,0,1,20220101,20221231\n",
}

JUNCTIONS_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" version="1" lat="43.6450" lon="-79.3500"><tag k="highway" v="motorway_junction"/></node>
  <node id="2" version="1" lat="43.6460" lon="-79.3790"><tag k="highway" v="traffic_signals"/></node>
</osm>
"""

NO_JUNCTION_OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="2" version="1" lat="43.6460" lon="-79.3790"><tag k="highway" v="traffic_signals"/></node>
</osm>
"""


@pytest.fixture
def gtfs(tmp_path: Path) -> GTFSTables:
    feed_dir = tmp_path / "feed"
    feed_dir.mkdir()
    for name, text in FEED.items():
        (feed_dir / name).write_text(text)

    return read_gtfs(GTFSFeeds(service_date=SERVICE_DATE, feeds={"ttc": "feed"}), tmp_path)


@pytest.fixture
def zones() -> gpd.GeoDataFrame:
    """Two zones of 0.01 degrees: ``west`` holds Union and bus stop A, ``east`` holds bus stops B and C."""
    zone_boxes = [box(-79.385, 43.640, -79.375, 43.650), box(-79.375, 43.640, -79.355, 43.650)]
    zone_ids = pd.Index(["west", "east"], name="loc_id")
    polygons = gpd.GeoSeries(zone_boxes, index=zone_ids, crs=CRS)
    projected_polygons = polygons.to_crs(UTM_CRS)

    return gpd.GeoDataFrame(
        {"original_geometry": projected_polygons}, geometry=projected_polygons.centroid, crs=UTM_CRS
    )


@pytest.fixture
def junctions(tmp_path: Path) -> gpd.GeoSeries:
    osm_path = tmp_path / "junctions.osm"
    osm_path.write_text(JUNCTIONS_OSM)

    return read_motorway_junctions(osm_path)


class TestRailStations:
    def test_are_the_parent_stations_of_the_rail_and_metro_stops_served_on_the_date(self, gtfs):
        stations = rail_stations(gtfs)

        assert stations["station_id"].tolist() == ["ttc:KING", "ttc:UNION"]

    def test_sit_at_the_parent_station(self, gtfs):
        stations = rail_stations(gtfs).set_index("station_id")

        assert stations.loc["ttc:UNION"].geometry.y == pytest.approx(43.6450)


class TestPeakDepartures:
    def test_count_the_peak_departures_of_every_mode_on_the_date(self, gtfs):
        departures = peak_departures(gtfs)
        counts = dict(zip(departures["stop_id"], departures["n_departures"], strict=True))

        assert counts == {"ttc:UNION_P1": 1, "ttc:BUS_B": 1}

    def test_interpolate_a_call_without_a_time_along_its_trip(self, gtfs):
        stop_ids = peak_departures(gtfs)["stop_id"].to_list()

        assert "ttc:BUS_B" in stop_ids

    def test_leave_out_the_last_call_of_a_trip(self, gtfs):
        stop_ids = peak_departures(gtfs)["stop_id"].to_list()

        assert "ttc:KING" not in stop_ids
        assert "ttc:BUS_C" not in stop_ids


class TestAddAccessFeatures:
    def test_add_the_access_columns(self, zones, gtfs, junctions):
        with_access = add_access_features(zones, gtfs, junctions)

        assert set(ACCESS_SCHEMA.names) <= set(with_access.columns)
        assert with_access.index.tolist() == ["west", "east"]

    def test_flag_the_zone_holding_a_station(self, zones, gtfs, junctions):
        with_access = add_access_features(zones, gtfs, junctions)

        assert with_access["has_rail_station"].tolist() == [1, 0]

    def test_measure_the_station_distance_from_the_zone_point(self, zones, gtfs, junctions):
        with_access = add_access_features(zones, gtfs, junctions)
        station_points = rail_stations(gtfs).to_crs(UTM_CRS).geometry
        expected = [station_points.distance(point).min() for point in zones.geometry]

        np.testing.assert_allclose(with_access["rail_station_distance_m"], expected)
        assert (with_access["rail_station_distance_m"] > 0).all()

    def test_count_the_peak_departures_per_hour_inside_each_zone(self, zones, gtfs, junctions):
        with_access = add_access_features(zones, gtfs, junctions)

        assert with_access["peak_departures_per_hour"].tolist() == [0.5, 0.5]

    def test_measure_the_junction_distance_from_the_zone_point(self, zones, gtfs, junctions):
        with_access = add_access_features(zones, gtfs, junctions)
        junction_point = junctions.to_crs(UTM_CRS).iloc[0]
        expected = [junction_point.distance(point) for point in zones.geometry]

        np.testing.assert_allclose(with_access["motorway_junction_distance_m"], expected)


class TestReadMotorwayJunctions:
    def test_reads_only_the_junction_nodes(self, junctions):
        assert len(junctions) == 1
        assert junctions.iloc[0].x == pytest.approx(-79.35)

    def test_an_extract_without_a_junction_raises(self, tmp_path):
        osm_path = tmp_path / "empty.osm"
        osm_path.write_text(NO_JUNCTION_OSM)

        with pytest.raises(ValueError, match="does not contain motorway junctions"):
            read_motorway_junctions(osm_path)
