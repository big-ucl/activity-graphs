"""Unit tests for the TTS loader (code mappings, builders, cohort filters), on synthetic frames of both survey versions."""

import city2graph as c2g
import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon, box

from activitygraphs.base import CRS, USER_JOURNEY_SCHEMA, USER_SCHEMA, Purpose
from activitygraphs.surveys.base import NA
from activitygraphs.surveys.demographics import DEMOGRAPHICS_SCHEMA
from activitygraphs.surveys.statistics import overlap_counts
from activitygraphs.surveys.tts import (
    CENSUS_2016_POPULATION,
    CENSUS_2021_POPULATION,
    DA_ID_COLUMN,
    PURPOSE_MAP_2016,
    PURPOSE_MAP_2022,
    REGION_COLUMN,
    ZONE_ID_COLUMN,
    TTSStatistics,
    TTSTables,
    _add_tts_population_statistics,
    _build_tts_island_edges,
    _build_tts_journeys,
    _build_tts_locations,
    _build_tts_survey,
    _build_tts_users,
    _read_census_population,
)
from activitygraphs.tensors.build import _build_group_ids, _build_home_nodes, _build_visit_labels

GTHA_ZONES = [1, 625, 1001, 5253]
GTHA_ZONE_REGIONS = [1, 1, 2, 6]
OUTSIDE_GTHA_ZONE = 6001
OUTSIDE_GTHA_REGION = 11
SPECIAL_ZONE_CODES = [0, 8888, 9998, 9999]
IDENTIFIER_COLUMNS = {"user_id", "hh_id", "home_loc_id"}


def _square(index: int) -> Polygon:
    """A small square near Toronto, offset so that every zone gets a distinct centroid."""
    lon, lat = -79.6 + 0.05 * index, 43.6 + 0.05 * index

    return Polygon([(lon, lat), (lon + 0.02, lat), (lon + 0.02, lat + 0.02), (lon, lat + 0.02)])


@pytest.fixture
def boundaries_zones() -> gpd.GeoDataFrame:
    zones = [*GTHA_ZONES, OUTSIDE_GTHA_ZONE]
    regions = [*GTHA_ZONE_REGIONS, OUTSIDE_GTHA_REGION]

    return gpd.GeoDataFrame(
        {ZONE_ID_COLUMN: zones, REGION_COLUMN: regions, "geometry": [_square(i) for i in range(len(zones))]},
        crs=CRS,
    )


def make_trips(rows: list[dict], survey_version: int) -> pl.DataFrame:
    purposes = (
        {"purp_orig": "H", "purp_dest": "W"} if survey_version == 2016 else {"purp_orig2022": 80, "purp_dest2022": 10}
    )
    defaults = {
        "hp_num": 1,
        "hpt_num": 1,
        "start_time": 830,
        "mode_prime": "D",
        **purposes,
        "gta06_orig": GTHA_ZONES[0],
        "gta06_dest": GTHA_ZONES[1],
        "trip_km": 5,
    }

    return pl.DataFrame([defaults | row for row in rows])


def make_persons(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "hhld_num": 1,
        "hp_num": 1,
        "age_range": "25-34",
        "driver_lic": "Y",
        "tran_pass": "N",
        "emp_stat": "F",
        "stu_stat": "O",
    }

    return pl.DataFrame([defaults | row for row in rows])


def make_households(rows: list[dict]) -> pl.DataFrame:
    defaults = {"hhld_num": 1, "gta06_hhld": GTHA_ZONES[0], "income": 4, "n_vehicle": 2, "trip_day": 2}

    return pl.DataFrame([defaults | row for row in rows])


def make_tables(
    boundaries_zones: gpd.GeoDataFrame,
    trips: pl.DataFrame | None = None,
    persons: pl.DataFrame | None = None,
    households: pl.DataFrame | None = None,
    survey_version: int = 2016,
) -> TTSTables:
    trips = make_trips([{}], survey_version) if trips is None else trips
    persons = make_persons([{}]) if persons is None else persons
    households = make_households([{}]) if households is None else households

    return TTSTables(trips, persons, households, boundaries_zones, survey_version)


@pytest.fixture
def locations_gdf(boundaries_zones) -> gpd.GeoDataFrame:
    return _build_tts_locations(make_tables(boundaries_zones))


class TestCodeMappings:
    def test_both_survey_versions_code_the_same_purposes(self):
        assert set(PURPOSE_MAP_2016.values()) == set(PURPOSE_MAP_2022.values())

    @pytest.mark.parametrize(("code_2016", "codes_2022"), [("M", [40, 41, 42, 44]), ("F", [51, 52]), ("D", [30])])
    def test_2022_codes_collapse_onto_the_2016_letter(self, code_2016, codes_2022):
        assert {PURPOSE_MAP_2022[code] for code in codes_2022} == {PURPOSE_MAP_2016[code_2016]}


class TestBuildLocations:
    def test_keeps_only_the_gtha_zones(self, locations_gdf):
        zones = locations_gdf[locations_gdf["type"] == "zone"]

        assert sorted(zones["loc_id"]) == sorted(str(zone) for zone in GTHA_ZONES)

    def test_the_region_column_picks_the_zones(self):
        zones = gpd.GeoDataFrame(
            {ZONE_ID_COLUMN: [1, 2], REGION_COLUMN: [1, OUTSIDE_GTHA_REGION], "geometry": [_square(0), _square(1)]},
            crs=CRS,
        )

        locations = _build_tts_locations(make_tables(zones))

        assert sorted(locations[locations["type"] == "zone"]["loc_id"]) == ["1"]

    def test_invalid_zone_geometries_are_repaired_into_polygons(self):
        spike = Polygon([(0, 0), (2, 0), (2, 2), (1, 2), (1, 4), (1, 2), (0, 2)])
        zones = gpd.GeoDataFrame({ZONE_ID_COLUMN: [1], REGION_COLUMN: [1], "geometry": [spike]}, crs="EPSG:26917")

        locations = _build_tts_locations(make_tables(zones))
        zone = locations[locations["type"] == "zone"]

        assert zone.is_valid.all()
        assert zone.geom_type.tolist() == ["Polygon"]

    def test_includes_the_na_sink(self, locations_gdf):
        assert NA in set(locations_gdf["loc_id"])


class TestBuildJourneys:
    @pytest.mark.parametrize("survey_version", [2016, 2022])
    def test_output_matches_the_journey_schema(self, survey_version, boundaries_zones, locations_gdf):
        tables = make_tables(boundaries_zones, trips=make_trips([{}], survey_version), survey_version=survey_version)

        journeys = _build_tts_journeys(tables, locations_gdf)

        assert journeys.schema == USER_JOURNEY_SCHEMA

    @pytest.mark.parametrize("destination", [*SPECIAL_ZONE_CODES, OUTSIDE_GTHA_ZONE])
    def test_unusable_destinations_become_the_na_sink(self, destination, boundaries_zones, locations_gdf):
        tables = make_tables(boundaries_zones, trips=make_trips([{"gta06_dest": destination}], 2016))

        journeys = _build_tts_journeys(tables, locations_gdf)

        assert journeys["arr_loc_id"].to_list() == [NA]

    def test_start_times_past_midnight_wrap_to_the_next_day(self, boundaries_zones, locations_gdf):
        trips = make_trips([{"hpt_num": 1, "start_time": 2530}, {"hpt_num": 2, "start_time": 9999}], 2016)

        journeys = _build_tts_journeys(make_tables(boundaries_zones, trips=trips), locations_gdf)

        assert [str(time) for time in journeys["dep_time"].to_list()] == ["01:30:00", "None"]

    def test_the_survey_versions_map_shopping_to_the_same_purpose(self, boundaries_zones, locations_gdf):
        trips_2016 = make_trips([{"purp_dest": "M"}], 2016)
        trips_2022 = make_trips([{"purp_dest2022": 44}], 2022)

        journeys_2016 = _build_tts_journeys(make_tables(boundaries_zones, trips=trips_2016), locations_gdf)
        journeys_2022 = _build_tts_journeys(
            make_tables(boundaries_zones, trips=trips_2022, survey_version=2022), locations_gdf
        )

        assert journeys_2016["arr_purpose"].to_list() == journeys_2022["arr_purpose"].to_list() == [Purpose.SHOP]

    def test_unmapped_codes_become_unknown(self, boundaries_zones, locations_gdf):
        trips = make_trips([{"purp_dest2022": 555, "mode_prime": "Z"}], 2022)

        journeys = _build_tts_journeys(make_tables(boundaries_zones, trips=trips, survey_version=2022), locations_gdf)

        assert journeys["arr_purpose"].to_list() == [Purpose.UNKNOWN]
        assert journeys["leg_mode"].to_list() == ["mode_unknown"]


class TestBuildUsers:
    def _build(self, boundaries_zones, locations_gdf, persons, households, trips=None) -> pl.DataFrame:
        tables = make_tables(boundaries_zones, trips=trips, persons=persons, households=households)
        journeys = _build_tts_journeys(tables, locations_gdf)

        return _build_tts_users(tables, locations_gdf, journeys)

    def test_output_matches_the_user_schema(self, boundaries_zones, locations_gdf):
        users = self._build(boundaries_zones, locations_gdf, make_persons([{}]), make_households([{}]))

        assert dict(users.schema) == dict(USER_SCHEMA)

    def test_maps_the_codes_of_both_survey_versions(self, boundaries_zones, locations_gdf):
        persons = make_persons([
            {"hp_num": 1, "age_range": "45-55", "emp_stat": "H", "tran_pass": "C"},
            {"hp_num": 2, "age_range": "75-99", "emp_stat": "J", "stu_stat": "P", "driver_lic": "0", "tran_pass": "X"},
            {"hp_num": 3, "emp_stat": "9", "stu_stat": "9", "driver_lic": "9", "tran_pass": "9"},
        ])
        trips = make_trips([{"hp_num": 1}, {"hp_num": 2}, {"hp_num": 3}], 2016)

        users = self._build(boundaries_zones, locations_gdf, persons, make_households([{}]), trips)

        assert users["age_band"].to_list() == ["45-54", "75+", "25-34"]
        assert users["employment_status"].to_list() == ["full_time", "part_time", None]
        assert users["student_status"].to_list() == ["not_student", "part_time", None]
        assert users["can_drive"].to_list() == [True, False, None]
        assert users["has_transit_pass"].to_list() == [True, False, None]

    def test_counts_every_household_member_including_those_without_trips(self, boundaries_zones, locations_gdf):
        persons = make_persons([
            {"hp_num": 1},
            {"hp_num": 2, "age_range": "13-17"},
            {"hp_num": 3, "age_range": "0-12"},
            {"hp_num": 4, "age_range": "65-74"},
        ])

        users = self._build(boundaries_zones, locations_gdf, persons, make_households([{}]))

        assert users["user_id"].to_list() == ["1"]
        assert users["hh_num_adults"].to_list() == [2]
        assert users["hh_num_children"].to_list() == [2]

    def test_unknown_vehicle_count_and_declined_income_become_missing(self, boundaries_zones, locations_gdf):
        persons = make_persons([{"hhld_num": 1, "hp_num": 1}, {"hhld_num": 2, "hp_num": 2}])
        households = make_households([{"hhld_num": 1, "n_vehicle": 99, "income": 7}, {"hhld_num": 2}])
        trips = make_trips([{"hp_num": 1, "hpt_num": 1}, {"hp_num": 2, "hpt_num": 2}], 2016)

        users = self._build(boundaries_zones, locations_gdf, persons, households, trips)

        assert users["hh_num_vehicles"].to_list() == [None, "2"]
        assert users["income_quartile"].to_list() == [None, "q3"]

    def test_income_quartiles_are_taken_over_the_gtha_households(self, boundaries_zones, locations_gdf):
        persons = make_persons([{"hhld_num": hh, "hp_num": hh} for hh in range(1, 5)])
        households = make_households([
            {"hhld_num": 1, "income": 1},
            {"hhld_num": 2, "income": 2},
            {"hhld_num": 3, "income": 6},
            {"hhld_num": 4, "income": 1, "gta06_hhld": OUTSIDE_GTHA_ZONE},
        ])
        trips = make_trips([{"hp_num": hh, "hpt_num": hh} for hh in range(1, 5)], 2016)

        users = self._build(boundaries_zones, locations_gdf, persons, households, trips)

        assert users["income_quartile"].to_list() == ["q1", "q3", "q4", "q1"]

    def test_outside_gtha_home_becomes_na(self, boundaries_zones, locations_gdf):
        persons = make_persons([{"hhld_num": 1, "hp_num": 1}, {"hhld_num": 2, "hp_num": 2}])
        households = make_households([{"hhld_num": 1, "gta06_hhld": OUTSIDE_GTHA_ZONE}, {"hhld_num": 2}])
        trips = make_trips([{"hp_num": 1, "hpt_num": 1}, {"hp_num": 2, "hpt_num": 2}], 2016)

        users = self._build(boundaries_zones, locations_gdf, persons, households, trips)

        assert users["home_loc_id"].to_list() == [NA, str(GTHA_ZONES[0])]

    def test_drops_users_whose_every_endpoint_is_the_na_sink(self, boundaries_zones, locations_gdf):
        persons = make_persons([{"hp_num": 1}, {"hp_num": 2}])
        trips = make_trips(
            [{"hp_num": 1}, {"hp_num": 2, "hpt_num": 2, "gta06_orig": 9999, "gta06_dest": OUTSIDE_GTHA_ZONE}], 2016
        )

        users = self._build(boundaries_zones, locations_gdf, persons, make_households([{}]), trips)

        assert users["user_id"].to_list() == ["1"]


@pytest.fixture
def tts_survey(boundaries_zones):
    persons = make_persons([
        {"hhld_num": 1, "hp_num": 1},
        {"hhld_num": 1, "hp_num": 2},
        {"hhld_num": 2, "hp_num": 3},
        {"hhld_num": 3, "hp_num": 4},
    ])
    households = make_households([
        {"hhld_num": 1, "gta06_hhld": GTHA_ZONES[0]},
        {"hhld_num": 2, "gta06_hhld": GTHA_ZONES[1]},
        {"hhld_num": 3, "gta06_hhld": OUTSIDE_GTHA_ZONE},
    ])
    trips = make_trips(
        [
            {"hp_num": 1, "hpt_num": 1, "gta06_orig": GTHA_ZONES[0], "gta06_dest": GTHA_ZONES[2]},
            {"hp_num": 2, "hpt_num": 2, "gta06_orig": GTHA_ZONES[0], "gta06_dest": GTHA_ZONES[2]},
            {"hp_num": 3, "hpt_num": 3, "gta06_orig": GTHA_ZONES[1], "gta06_dest": GTHA_ZONES[2]},
            {"hp_num": 4, "hpt_num": 4, "gta06_orig": OUTSIDE_GTHA_ZONE, "gta06_dest": GTHA_ZONES[2]},
        ],
        2016,
    )
    tables = make_tables(boundaries_zones, trips=trips, persons=persons, households=households)

    return _build_tts_survey(tables).with_filter("zone")


class TestTTSSurveyViews:
    def test_users_without_a_gtha_home_are_filtered_out(self, tts_survey):
        assert tts_survey.users_df["user_id"].to_list() == ["1", "2", "3"]

    def test_every_users_df_column_is_a_feature_or_an_identifier(self, tts_survey):
        assert set(tts_survey.demographics_schema.names) | IDENTIFIER_COLUMNS == set(tts_survey.users_df.columns)

    def test_demographics_are_the_shared_schema(self, tts_survey):
        assert tts_survey.demographics_schema == DEMOGRAPHICS_SCHEMA

    def test_group_ids_are_shared_by_co_residents_only(self, tts_survey):
        group_ids = _build_group_ids(tts_survey)

        assert group_ids[0] == group_ids[1]
        assert group_ids[2] != group_ids[0]

    def test_home_nodes_are_the_users_home_zones(self, tts_survey):
        """The nodes are in ``loc_id`` string order: zones 1, 1001, 5253, 625."""
        assert _build_home_nodes(tts_survey).tolist() == [0, 0, 3]

    def test_visit_labels_mark_both_trip_ends(self, tts_survey):
        labels = _build_visit_labels(tts_survey, list(Purpose))

        assert labels.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]]


UTM_17N = "EPSG:26917"
TORONTO_UTM = (630_000, 4_835_000)


def utm_zones(loc_ids: list[str], boxes: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"loc_id": loc_ids, "geometry": [box(*bounds) for bounds in boxes]}, crs=UTM_17N)


class TestPopulationStatistics:
    def test_each_area_is_split_by_its_share_of_area_in_the_zone(self):
        zones = utm_zones(["a", "b", "c"], [(0, 0, 100, 100), (100, 0, 200, 100), (500, 500, 600, 600)])
        areas = gpd.GeoDataFrame(
            {"population": [100.0, 40.0], "geometry": [box(50, 0, 150, 100), box(100, 0, 200, 100)]}, crs=UTM_17N
        )

        counts = overlap_counts(zones, areas, "population")

        assert counts.index.tolist() == ["a", "b", "c"]
        assert counts.tolist() == pytest.approx([50.0, 90.0, 0.0])

    def test_adds_the_population_density_of_each_zone(self):
        x, y = TORONTO_UTM
        zones = utm_zones(["a"], [(x, y, x + 100, y + 100)])
        areas = gpd.GeoDataFrame({"population": [200.0], "geometry": [box(x, y, x + 100, y + 200)]}, crs=UTM_17N)

        located = _add_tts_population_statistics(zones, TTSStatistics(areas))

        assert located["population"].tolist() == pytest.approx([100.0 / 10_000])
        assert "jobs" not in located.columns

    @pytest.mark.parametrize("census", [CENSUS_2016_POPULATION, CENSUS_2021_POPULATION])
    def test_reads_the_population_rows_of_each_census_format(self, census, tmp_path):
        rows = [
            {census.da_id_col: "35200001", census.characteristic_col: "1", census.value_col: "512"},
            {census.da_id_col: "35200001", census.characteristic_col: "2", census.value_col: "9"},
            {census.da_id_col: "35200002", census.characteristic_col: "1", census.value_col: "x"},
        ]
        path = tmp_path / "profile.csv"
        pl.DataFrame(rows).write_csv(path)

        population = _read_census_population(path, census)

        assert population[DA_ID_COLUMN].to_list() == ["35200001", "35200002"]
        assert population["population"].to_list() == [512.0, 0.0]


class TestIslandEdges:
    def _graph(self, boxes: dict[str, tuple[float, float, float, float]]):
        zones = utm_zones(list(boxes), list(boxes.values())).set_index("loc_id")

        return c2g.contiguity_graph(zones, set_point_nodes=True)

    def test_joins_a_zone_without_neighbours_to_its_nearest_zones(self):
        x, y = TORONTO_UTM
        nodes, edges = self._graph({
            "1": (x, y, x + 100, y + 100),
            "2": (x + 100, y, x + 200, y + 100),
            "3": (x + 200, y, x + 300, y + 100),
            "4": (x + 300, y, x + 400, y + 100),
            "81": (x, y + 300, x + 100, y + 400),
        })

        island_edges = _build_tts_island_edges(nodes, edges)

        assert sorted(island_edges.index.tolist()) == [("81", "1"), ("81", "2"), ("81", "3")]

    def test_adds_no_edge_when_every_zone_has_a_neighbour(self):
        x, y = TORONTO_UTM
        nodes, edges = self._graph({"1": (x, y, x + 100, y + 100), "2": (x + 100, y, x + 200, y + 100)})

        assert _build_tts_island_edges(nodes, edges).empty
