"""Tests of the GTFS reader on two small feeds, and of the feeds configured for each dataset."""

import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from activitygraphs.config import GTFSFeeds
from activitygraphs.surveys.gtfs import PTMode, read_gtfs

SERVICE_DATE = "2022-05-18"

EXTENDED_FEED = {
    "stops.txt": '﻿"stop_id","stop_name","stop_lat","stop_lon","location_type","parent_station"\n'
    '"P1","Central","46.2","6.1","1",""\n'
    '"1","Central A","46.2","6.1","","P1"\n'
    '"2","Lakeside","46.21","6.12","",""\n',
    "routes.txt": '"route_id","agency_id","route_short_name","route_long_name","route_desc","route_type"\n'
    '"S1","11","SL1","","S","109"\n'
    '"M1","11","M1","","","401"\n'
    '"B1","12","1","","B","700"\n'
    '"T1","12","12","","T","900"\n'
    '"F1","13","","Lake","","1000"\n'
    '"C1","14","","Cable car","","1300"\n',
    "trips.txt": '"route_id","service_id","trip_id","trip_headsign"\n'
    '"S1","WEEKLY","t-weekly","Central"\n'
    '"B1","EXPIRED","t-expired","Central"\n'
    '"T1","TUESDAY","t-tuesday","Central"\n'
    '"F1","ADDED","t-added","Central"\n'
    '"M1","REMOVED","t-removed","Central"\n',
    "stop_times.txt": '"trip_id","arrival_time","departure_time","stop_id","stop_sequence"\n'
    '"t-weekly","8:05:00","8:05:30","1","1"\n'
    '"t-weekly","25:10:00","25:10:00","2","2"\n',
    "calendar.txt": '"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday",'
    '"start_date","end_date"\n'
    '"WEEKLY","1","1","1","1","1","0","0","20220101","20221231"\n'
    '"EXPIRED","1","1","1","1","1","0","0","20210101","20211231"\n'
    '"TUESDAY","0","1","0","0","0","0","0","20220101","20221231"\n'
    '"REMOVED","1","1","1","1","1","0","0","20220101","20221231"\n',
    "calendar_dates.txt": '"service_id","date","exception_type"\n'
    '"ADDED","20220518","1"\n'
    '"REMOVED","20220518","2"\n'
    '"TUESDAY","20220519","1"\n',
}

BASIC_FEED = {
    "stops.txt": "stop_id, stop_name, stop_desc, stop_lat, stop_lon\r\n"
    "UNION, Union Station, , 41.8786, -87.6403\r\n"
    "OGILVIE, Ogilvie, , 41.8826, -87.6402\r\n",
    "routes.txt": "route_id, route_short_name, route_long_name, route_type\r\n"
    "RED, Red, Red Line, 1\r\n"
    "BNSF, BNSF, Burlington, 2\r\n"
    "STC, 501, Queen, 0\r\n"
    "BUS9, 9, Ashland, 3\r\n",
    "trips.txt": "route_id, service_id, trip_id\r\nBNSF, A1, BN1200\r\nRED, A2, R1\r\n",
    "stop_times.txt": "trip_id, arrival_time, departure_time, stop_id, stop_sequence\r\n"
    "BN1200,  04:14:00,  04:14:00,  UNION,  1\r\n",
    "calendar_dates.txt": "service_id, date, exception_type\r\nA1, 20220518, 1\r\nA2, 20220519, 1\r\n",
}


@pytest.fixture
def feeds(tmp_path: Path) -> GTFSFeeds:
    extended_dir = tmp_path / "extended"
    extended_dir.mkdir()
    for name, text in EXTENDED_FEED.items():
        (extended_dir / name).write_text(text, encoding="utf-8")

    with zipfile.ZipFile(tmp_path / "basic.zip", "w") as archive:
        for name, text in BASIC_FEED.items():
            archive.writestr(f"feed/{name}", text)

    return GTFSFeeds(service_date=SERVICE_DATE, feeds={"ext": "extended", "basic": "basic.zip"})


def test_ids_are_prefixed_by_feed(feeds, tmp_path):
    stops = read_gtfs(feeds, tmp_path).stops_df
    parents = dict(zip(stops["stop_id"], stops["parent_station"], strict=True))

    assert parents == {
        "ext:P1": None,
        "ext:1": "ext:P1",
        "ext:2": None,
        "basic:UNION": None,
        "basic:OGILVIE": None,
    }


def test_padded_fields_and_absent_optional_columns_are_read(feeds, tmp_path):
    stops = read_gtfs(feeds, tmp_path).stops_df.filter(pl.col("feed") == "basic")

    assert stops["stop_name"].to_list() == ["Union Station", "Ogilvie"]
    assert stops["stop_lat"].to_list() == [41.8786, 41.8826]
    assert stops["location_type"].null_count() == stops.height


def test_basic_and_extended_route_types_map_to_one_mode_set(feeds, tmp_path):
    routes = read_gtfs(feeds, tmp_path).routes_df
    modes = dict(zip(routes["route_id"], routes["mode"], strict=True))

    assert modes == {
        "ext:S1": PTMode.RAIL,
        "ext:M1": PTMode.METRO,
        "ext:B1": PTMode.BUS,
        "ext:T1": PTMode.TRAM,
        "ext:F1": PTMode.FERRY,
        "ext:C1": PTMode.OTHER,
        "basic:RED": PTMode.METRO,
        "basic:BNSF": PTMode.RAIL,
        "basic:STC": PTMode.TRAM,
        "basic:BUS9": PTMode.BUS,
    }


def test_times_are_seconds_after_midnight(feeds, tmp_path):
    stop_times = read_gtfs(feeds, tmp_path).stop_times_df.collect()
    times = dict(zip(stop_times["stop_id"], stop_times["arrival_time"], strict=True))

    assert times == {"ext:1": 8 * 3600 + 5 * 60, "ext:2": 25 * 3600 + 10 * 60, "basic:UNION": 4 * 3600 + 14 * 60}


def test_trips_on_service_date_follow_calendar_and_exceptions(feeds, tmp_path):
    gtfs = read_gtfs(feeds, tmp_path)
    trips = gtfs.trips_on_service_date().collect()

    assert gtfs.service_date == date(2022, 5, 18)
    assert sorted(trips["trip_id"]) == ["basic:BN1200", "ext:t-added", "ext:t-weekly"]


def test_unknown_route_type_raises(feeds, tmp_path):
    routes = tmp_path / "extended" / "routes.txt"
    routes.write_text('"route_id","route_type"\n"X","13"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="route type"):
        read_gtfs(feeds, tmp_path)


def test_missing_required_column_raises(feeds, tmp_path):
    trips = tmp_path / "extended" / "trips.txt"
    trips.write_text('"route_id","trip_id"\n"S1","t-weekly"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="service_id"):
        read_gtfs(feeds, tmp_path)


@pytest.mark.parametrize("data", ["tpg", "cmap", "tts2016", "tts2022"])
def test_every_configured_feed_runs_trips_on_its_service_date(load_or_skip, data):
    def read(cfg, root):
        return read_gtfs(cfg.data.inputs.gtfs, root / cfg.data.paths.gtfs), set(cfg.data.inputs.gtfs.feeds)

    gtfs, configured = load_or_skip(data, read)
    trips_per_feed = gtfs.trips_on_service_date().group_by("feed").len().collect()

    assert set(trips_per_feed["feed"]) == configured
    assert trips_per_feed["len"].min() > 0
