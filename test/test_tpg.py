import re

import polars as pl
import pytest

from activitygraphs.surveys.tpg import LOCATION_REGEXES, _build_stop_names_to_loc_id_mapping


@pytest.fixture
def location_examples():
    return {
        "subsector": ["Aéroport - Arena - sous_secteur", "Pierres-du-Niton - sous_secteur"],
        "municipality_swiss": ["Genève - 1203"],
        "municipality_french": ["FERNEY-VOLTAIRE - 01210"],
        "na": ["NA"],
        "other": ["Genève-Cornavin"],
    }


@pytest.mark.parametrize("location_type", LOCATION_REGEXES.keys())
def test_location_regex_passes_own_example(location_type: str, location_examples: dict):
    for example in location_examples[location_type]:
        assert re.match(LOCATION_REGEXES[location_type], example)


@pytest.mark.parametrize("location_type", LOCATION_REGEXES.keys())
def test_location_regex_fails_other_examples(location_type: str, location_examples: dict[str, str]):
    regex = LOCATION_REGEXES[location_type]
    other_location_types = set(location_examples.keys()) - {location_type}

    for other_type in other_location_types:
        for other_example in location_examples[other_type]:
            assert not re.match(regex, other_example)


def test_each_stop_name_maps_to_one_id_preferring_the_parent_station():
    stops = pl.DataFrame(
        {
            "stop_id": ["s:Parent1", "s:1:0:A", "s:2", "s:2:0:R", "s:3", "s:Parent4", "s:4b"],
            "stop_name": ["Central", "Central", "Lakeside", "lakeside", "Hill", "Market", "Market"],
            "location_type": [1, None, None, None, None, 1, None],
            "parent_station": [None, "s:Parent1", None, None, None, None, None],
        },
        schema_overrides={"location_type": pl.Int8},
    )

    mapping = _build_stop_names_to_loc_id_mapping(stops)

    assert dict(zip(mapping["loc_name"].str.to_lowercase(), mapping["loc_id"], strict=True)) == {
        "central": "s:Parent1",
        "lakeside": "s:2",
        "hill": "s:3",
        "market": "s:Parent4",
    }
