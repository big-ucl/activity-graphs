import re

import pytest

from activitygraphs.surveys.tpg import LOCATION_REGEXES


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
