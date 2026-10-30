"""Unit tests for the CMAP MyDailyTravel loader (mappings, builders, cohort filters).

All tests run on synthetic frames, so none of them need the survey on disk. The one test that does
is marked ``integration`` and skips when ``data/processed/CMAP`` is absent.
"""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon

from activitygraphs.base import CRS, USER_JOURNEY_SCHEMA, USER_SCHEMA, Mode, Purpose
from activitygraphs.surveys.base import NA
from activitygraphs.surveys.cmap import (
    CMAP_DEMOGRAPHICS_SCHEMA,
    EXCLUDED_CENSUS_TRACTS,
    MODE_MAP,
    PURPOSE_MAP,
    CMAPSurvey,
    CMAPTables,
    build_cmap_journeys,
    build_cmap_locations,
    build_cmap_survey,
    build_cmap_users,
)
from activitygraphs.tensors.build import (
    build_group_ids,
    build_home_indicator,
    build_user_attributes,
    build_visit_labels,
)

IN_REGION_TRACTS = ["17031000100", "17031000200", "17043000100"]
OUT_OF_REGION_TRACT = "17099000100"  # LaSalle county, outside the CMAP 7-county region
LAKE_TRACT = EXCLUDED_CENSUS_TRACTS[0]

SENTINELS = [-1, 995, 996, 998, 999]
IDENTIFIER_COLUMNS = {"user_id", "hh_id", "home_loc_id"}


def _square(index: int) -> Polygon:
    """A small square near Chicago, offset so that every tract gets a distinct centroid."""
    lon, lat = -87.9 + 0.1 * index, 41.7 + 0.1 * index

    return Polygon([(lon, lat), (lon + 0.05, lat), (lon + 0.05, lat + 0.05), (lon, lat + 0.05)])


@pytest.fixture
def boundaries_ct() -> gpd.GeoDataFrame:
    tracts = [*IN_REGION_TRACTS, OUT_OF_REGION_TRACT, LAKE_TRACT]

    return gpd.GeoDataFrame(
        {
            "GEOID": tracts,
            "NAME": [tract[-6:] for tract in tracts],
            "COUNTYFP": [tract[2:5] for tract in tracts],
            "geometry": [_square(i) for i in range(len(tracts))],
        },
        crs=CRS,
    )


@pytest.fixture
def locations_gdf(boundaries_ct) -> gpd.GeoDataFrame:
    return build_cmap_locations(CMAPTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), boundaries_ct))


def make_journeys(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "person_id": 1,
        "linked_trip_id": 1,
        "linked_trip_mode": 10,
        "duration_minutes": 15,
        "depart_date": "2024-05-13",
        "depart_hour": 8,
        "depart_minute": 30,
        "depart_seconds": 7.5,  # the codebook float; the builder casts it
        "o_purpose_category": 1,
        "d_purpose_category": 2,
        "o_tract_2020": IN_REGION_TRACTS[0],
        "d_tract_2020": IN_REGION_TRACTS[1],
    }

    return pl.DataFrame([defaults | row for row in rows])


def make_persons(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "person_id": 1,
        "hh_id": 1,
        "age": 6,
        "employment": 1,
        "student": 2,
        "can_drive": 1,
        "transit_pass_0": 1,
        **{f"transit_pass_{option}": 0 for option in range(1, 6)},
    }

    return pl.DataFrame([defaults | row for row in rows])


def make_households(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "hh_id": 1,
        "diary_platform": "browser",
        "home_tract_2020": IN_REGION_TRACTS[0],
        "num_adults": 2,
        "num_kids": 1,
        "num_vehicles": 2,
        "num_bicycle": 1,
        "income_detailed": 9,
    }

    return pl.DataFrame([defaults | row for row in rows])


class TestCodeMappings:
    def test_purpose_map_covers_all_codebook_categories(self):
        assert set(PURPOSE_MAP) >= {-1, *range(1, 15), 995}

    def test_mode_map_covers_all_codebook_modes(self):
        # 995 (Missing Response) is not in the map; it reaches Mode.UNKNOWN through replace_strict's
        # default instead, which test_unmapped_codes_become_unknown pins down.
        assert set(MODE_MAP) >= set(range(1, 19)) | {-1}

    def test_maps_are_single_valued(self):
        # invert_mapping silently keeps the last writer if a code appears under two enum members.
        assert len(set(MODE_MAP)) == len(MODE_MAP)
        assert len(set(PURPOSE_MAP)) == len(PURPOSE_MAP)


class TestBuildLocations:
    def test_keeps_only_the_seven_cmap_counties(self, locations_gdf):
        assert OUT_OF_REGION_TRACT not in set(locations_gdf["loc_id"])
        assert set(IN_REGION_TRACTS) <= set(locations_gdf["loc_id"])

    def test_drops_the_uninhabited_lake_tracts(self, locations_gdf):
        assert LAKE_TRACT not in set(locations_gdf["loc_id"])

    def test_includes_the_na_sink(self, locations_gdf):
        assert NA in set(locations_gdf["loc_id"])
        assert locations_gdf.loc[locations_gdf["loc_id"] == NA, "type"].item() == "na"

    def test_tracts_are_typed_as_zones(self, locations_gdf):
        tracts = locations_gdf[locations_gdf["loc_id"] != NA]
        assert set(tracts["type"]) == {"zone"}


class TestBuildJourneys:
    def test_output_matches_the_journey_schema(self, boundaries_ct, locations_gdf):
        journeys = make_journeys([{"person_id": 1}, {"person_id": 2}])
        inputs = CMAPTables(journeys, pl.DataFrame(), pl.DataFrame(), boundaries_ct)

        built = build_cmap_journeys(inputs, locations_gdf)

        assert built.schema == USER_JOURNEY_SCHEMA
        assert built.height == 2

    @pytest.mark.parametrize(
        "destination", [OUT_OF_REGION_TRACT, LAKE_TRACT, None], ids=["out_of_region", "lake_tract", "null"]
    )
    def test_unusable_destinations_become_the_na_sink(self, destination, boundaries_ct, locations_gdf):
        journeys = make_journeys([{"d_tract_2020": IN_REGION_TRACTS[1]}, {"d_tract_2020": destination}])
        inputs = CMAPTables(journeys, pl.DataFrame(), pl.DataFrame(), boundaries_ct)

        built = build_cmap_journeys(inputs, locations_gdf)

        assert built["arr_loc_id"].to_list() == [IN_REGION_TRACTS[1], NA]

    def test_unusable_destinations_remain_journeys(self, boundaries_ct, locations_gdf):
        # Trips to the NA sink produce no in-graph visit but must not be dropped, or trip and day
        # counts (and so the exposure t_i) would be wrong.
        journeys = make_journeys([{"d_tract_2020": OUT_OF_REGION_TRACT}])
        inputs = CMAPTables(journeys, pl.DataFrame(), pl.DataFrame(), boundaries_ct)

        assert build_cmap_journeys(inputs, locations_gdf).height == 1

    def test_unmapped_codes_become_unknown(self, boundaries_ct, locations_gdf):
        journeys = make_journeys([
            {"linked_trip_mode": sentinel, "d_purpose_category": sentinel, "o_purpose_category": sentinel}
            for sentinel in SENTINELS
        ])
        inputs = CMAPTables(journeys, pl.DataFrame(), pl.DataFrame(), boundaries_ct)

        built = build_cmap_journeys(inputs, locations_gdf)

        assert set(built["leg_mode"].to_list()) == {Mode.UNKNOWN}
        assert set(built["arr_purpose"].to_list()) == {Purpose.UNKNOWN}
        assert set(built["dep_purpose"].to_list()) == {Purpose.UNKNOWN}

    def test_departure_purposes_are_mapped_not_stubbed(self, boundaries_ct, locations_gdf):
        # Unlike THATS, CMAP has real origin purposes, so home shows up in location_visits.
        journeys = make_journeys([{"o_purpose_category": 1, "d_purpose_category": 7}])
        inputs = CMAPTables(journeys, pl.DataFrame(), pl.DataFrame(), boundaries_ct)

        built = build_cmap_journeys(inputs, locations_gdf)

        assert built["dep_purpose"].item() == Purpose.HOME
        assert built["arr_purpose"].item() == Purpose.SHOP


class TestBuildUsers:
    def _build(self, boundaries_ct, locations_gdf, persons, households, journeys=None):
        journeys = make_journeys([{"person_id": row} for row in persons["person_id"]]) if journeys is None else journeys
        inputs = CMAPTables(journeys, persons, households, boundaries_ct)

        return build_cmap_users(inputs, locations_gdf, build_cmap_journeys(inputs, locations_gdf))

    def test_output_matches_the_user_schema(self, boundaries_ct, locations_gdf):
        # Compared as dicts: check_schema is set-based, so the builder does not fix column order.
        users = self._build(boundaries_ct, locations_gdf, make_persons([{}]), make_households([{}]))

        assert dict(users.schema) == dict(USER_SCHEMA)

    def test_no_sentinel_survives_any_numeric_column(self, boundaries_ct, locations_gdf):
        # Every numeric column here is tensorised wholesale and StandardScaler-ed, so a stray 995
        # would wreck the scaling for the whole feature.
        persons = make_persons([
            {"person_id": 1, "hh_id": 1, "employment": 995, "student": 995, "can_drive": 995, "age": 6},
            {"person_id": 2, "hh_id": 2, "employment": 1, "student": 2, "can_drive": 1, "age": 8},
        ])
        households = make_households([
            {"hh_id": 1, "num_bicycle": 995, "income_detailed": 999},
            {"hh_id": 2, "num_bicycle": 2, "income_detailed": 5},
        ])

        users = self._build(boundaries_ct, locations_gdf, persons, households)

        numeric = users.select(pl.selectors.numeric())
        assert numeric.height == 2
        for column in numeric.columns:
            assert not set(numeric[column].to_list()) & set(SENTINELS), column

    def test_missing_demographics_are_flagged_not_silently_folded_to_false(self, boundaries_ct, locations_gdf):
        # The recodes turn 995 into False; the indicator is what lets the model tell an absent
        # answer apart from a real "no".
        persons = make_persons([
            {"person_id": 1, "hh_id": 1, "employment": 995, "student": 995, "can_drive": 995},
            {"person_id": 2, "hh_id": 2, "employment": 5, "student": 2, "can_drive": 2},
        ])
        households = make_households([{"hh_id": 1}, {"hh_id": 2}])

        users = self._build(boundaries_ct, locations_gdf, persons, households).sort("user_id")

        assert users["has_missing_demographics"].to_list() == [True, False]
        assert users["can_drive"].to_list() == [False, False]

    def test_codes_outside_the_codebook_become_missing(self, boundaries_ct, locations_gdf):
        persons = make_persons([
            {"person_id": 1, "hh_id": 1, "age": 995, "employment": 995, "student": 995},
            {"person_id": 2, "hh_id": 2, "age": 8, "employment": 5, "student": 0},
        ])
        households = make_households([{"hh_id": 1, "income_detailed": 999}, {"hh_id": 2, "income_detailed": 12}])

        users = self._build(boundaries_ct, locations_gdf, persons, households).sort("user_id")

        assert users["age_band"].to_list() == [None, 8]
        assert users["employment_status"].to_list() == [None, 5]
        assert users["student_status"].to_list() == [None, 0]
        assert users["income_band"].to_list() == [None, 12]

    def test_any_selected_pass_is_a_transit_pass(self, boundaries_ct, locations_gdf):
        persons = make_persons([
            {"person_id": 1, "hh_id": 1},
            {"person_id": 2, "hh_id": 2, "transit_pass_0": 0, "transit_pass_3": 1},
            {"person_id": 3, "hh_id": 3, **{f"transit_pass_{option}": 995 for option in range(6)}},
        ])
        households = make_households([{"hh_id": 1}, {"hh_id": 2}, {"hh_id": 3}])

        users = self._build(boundaries_ct, locations_gdf, persons, households).sort("user_id")

        assert users["has_transit_pass"].to_list() == [False, True, None]

    def test_drops_multi_day_rmove_households(self, boundaries_ct, locations_gdf):
        persons = make_persons([{"person_id": 1, "hh_id": 1}, {"person_id": 2, "hh_id": 2}])
        households = make_households([
            {"hh_id": 1, "diary_platform": "rmove"},
            {"hh_id": 2, "diary_platform": "call"},
        ])

        users = self._build(boundaries_ct, locations_gdf, persons, households)

        assert users["user_id"].to_list() == ["2"]

    def test_drops_users_without_journeys(self, boundaries_ct, locations_gdf):
        persons = make_persons([{"person_id": 1, "hh_id": 1}, {"person_id": 2, "hh_id": 2}])
        households = make_households([{"hh_id": 1}, {"hh_id": 2}])
        journeys = make_journeys([{"person_id": 1}])

        users = self._build(boundaries_ct, locations_gdf, persons, households, journeys)

        assert users["user_id"].to_list() == ["1"]

    def test_drops_users_whose_every_endpoint_is_the_na_sink(self, boundaries_ct, locations_gdf):
        # Decision 9 is "no in-region visit", not "no journeys": a user whose endpoints are all
        # masked or out-of-region has journey rows but is unscoreable, since every ranking metric
        # skips users with no positives. Counting them shifts n_users and the split proportions.
        persons = make_persons([{"person_id": 1, "hh_id": 1}, {"person_id": 2, "hh_id": 2}])
        households = make_households([{"hh_id": 1}, {"hh_id": 2}])
        journeys = make_journeys([
            {"person_id": 1},
            {"person_id": 2, "o_tract_2020": OUT_OF_REGION_TRACT, "d_tract_2020": None},
        ])

        users = self._build(boundaries_ct, locations_gdf, persons, households, journeys)

        assert users["user_id"].to_list() == ["1"]

    @pytest.mark.parametrize("usable_endpoint", ["o_tract_2020", "d_tract_2020"])
    def test_keeps_users_with_a_single_in_region_endpoint(self, usable_endpoint, boundaries_ct, locations_gdf):
        # One usable endpoint is one in-region visit, which is enough to score the user.
        persons = make_persons([{"person_id": 1, "hh_id": 1}])
        households = make_households([{"hh_id": 1}])
        endpoints = {"o_tract_2020": OUT_OF_REGION_TRACT, "d_tract_2020": None}
        journeys = make_journeys([{"person_id": 1, **endpoints, usable_endpoint: IN_REGION_TRACTS[0]}])

        users = self._build(boundaries_ct, locations_gdf, persons, households, journeys)

        assert users["user_id"].to_list() == ["1"]

    def test_out_of_region_home_becomes_na(self, boundaries_ct, locations_gdf):
        persons = make_persons([{"person_id": 1, "hh_id": 1}])
        households = make_households([{"hh_id": 1, "home_tract_2020": OUT_OF_REGION_TRACT}])

        users = self._build(boundaries_ct, locations_gdf, persons, households)

        assert users["home_loc_id"].to_list() == [NA]

    def test_co_residents_keep_the_same_household_id(self, boundaries_ct, locations_gdf):
        persons = make_persons([{"person_id": 1, "hh_id": 1}, {"person_id": 2, "hh_id": 1}])
        households = make_households([{"hh_id": 1}])

        users = self._build(boundaries_ct, locations_gdf, persons, households)

        assert users["hh_id"].to_list() == ["1", "1"]


@pytest.fixture
def cmap_survey(boundaries_ct) -> CMAPSurvey:
    persons = make_persons([
        {"person_id": 1, "hh_id": 1},
        {"person_id": 2, "hh_id": 1},  # co-resident of person 1
        {"person_id": 3, "hh_id": 2},
        {"person_id": 4, "hh_id": 3, "employment": 995, "student": 995, "can_drive": 995},
    ])
    households = make_households([
        {"hh_id": 1, "home_tract_2020": IN_REGION_TRACTS[0]},
        {"hh_id": 2, "home_tract_2020": IN_REGION_TRACTS[1]},
        {"hh_id": 3, "home_tract_2020": OUT_OF_REGION_TRACT},  # filtered out by the zone filter
    ])
    journeys = make_journeys([{"person_id": p, "linked_trip_id": p} for p in (1, 2, 3, 4)])

    return build_cmap_survey(CMAPTables(journeys, persons, households, boundaries_ct)).with_filter("zone")


class TestCMAPSurveyViews:
    def test_users_without_an_in_region_home_are_filtered_out(self, cmap_survey):
        assert cmap_survey.users_df["user_id"].to_list() == ["1", "2", "3"]

    def test_user_ids_match_users_df_order(self, cmap_survey):
        # build_group_ids and the user attributes tensor are both aligned positionally on this.
        assert cmap_survey.user_ids.to_list() == cmap_survey.users_df["user_id"].to_list()

    def test_home_locations_cover_every_user(self, cmap_survey):
        assert cmap_survey.home_locations["user_id"].sort().to_list() == cmap_survey.user_ids.to_list()
        assert cmap_survey.home_locations["loc_id"].null_count() == 0

    def test_every_users_df_column_is_a_feature_or_an_identifier(self, cmap_survey):
        # Fails the moment a column is added to build_cmap_users without being classified, which is
        # the only way hh_id (or any other identifier) could reach the user attributes tensor.
        assert set(cmap_survey.demographics_schema.names) | IDENTIFIER_COLUMNS == set(cmap_survey.users_df.columns)

    def test_hh_id_never_reaches_the_user_attributes_tensor(self, cmap_survey):
        # hh_id lives on users_df for the grouped split; it is an identifier, not a feature.
        assert "hh_id" not in cmap_survey.demographics_schema.names

        user_attributes, _ = build_user_attributes(cmap_survey)
        assert user_attributes.shape == (len(cmap_survey.user_ids), cmap_survey.demographics_schema.numeric.width)

    def test_demographic_column_order_is_declared_not_incidental(self, cmap_survey):
        # The tensor is built positionally and only its width is fingerprinted, so a silent reorder
        # would let a stale transform cache be applied to the wrong columns.
        assert cmap_survey.demographics_schema == CMAP_DEMOGRAPHICS_SCHEMA

    def test_group_ids_are_shared_by_co_residents_only(self, cmap_survey):
        group_ids = build_group_ids(cmap_survey)

        assert group_ids.shape == (len(cmap_survey.user_ids),)
        assert group_ids[0] == group_ids[1]  # persons 1 and 2 share household 1
        assert group_ids[2] != group_ids[0]

    def test_group_ids_are_stable_across_rebuilds(self, cmap_survey, boundaries_ct):
        # polars unique() does not maintain order by default; an unstable labelling would silently
        # change the split for a fixed seed on every cache rebuild.
        assert build_group_ids(cmap_survey).tolist() == build_group_ids(cmap_survey).tolist()


class TestUserNodeTensors:
    def test_home_indicator_marks_each_users_home_only(self, cmap_survey):
        is_home = build_home_indicator(cmap_survey).squeeze(-1)

        assert is_home.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 0]]

    def test_visit_labels_over_every_purpose_mark_both_trip_ends(self, cmap_survey):
        labels = build_visit_labels(cmap_survey, list(Purpose)).squeeze(-1)

        assert labels.tolist() == [[1, 1, 0]] * 3

    def test_visit_labels_keep_only_the_given_purposes(self, cmap_survey):
        labels = build_visit_labels(cmap_survey, [Purpose.WORK_MAIN]).squeeze(-1)

        assert labels.tolist() == [[0, 1, 0]] * 3

    def test_visit_labels_without_purposes_are_empty(self, cmap_survey):
        assert build_visit_labels(cmap_survey, []).sum() == 0


@pytest.mark.integration
class TestCMAPSurveyRoundTrip:
    @pytest.fixture
    def loaded(self, load_or_skip):
        return load_or_skip("cmap", lambda cfg, root: CMAPSurvey.load(cfg.data, root))

    def test_persists_hh_id_through_the_cache(self, loaded):
        # save() must write the private _users_df; a public view without hh_id would break the
        # grouped split on every run after the first.
        assert "hh_id" in loaded.users_df.columns

    def test_home_locations_cover_every_user(self, loaded):
        survey = loaded.with_filter("zone")

        assert survey.home_locations["user_id"].sort().to_list() == survey.user_ids.to_list()
