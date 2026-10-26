"""Unit tests for the survey registry."""

from typing import get_args

import pytest

from activitygraphs.config import VALID_CONFIGS, read_config
from activitygraphs.surveys.registry import REGISTRY, survey_loader
from activitygraphs.utils import get_project_root


@pytest.mark.parametrize("data", get_args(VALID_CONFIGS))
def test_every_data_config_names_a_registered_survey(data: str):
    cfg = read_config(get_project_root(), verbose=False, data=data)

    assert cfg.data.survey in REGISTRY


def test_an_unregistered_survey_raises(monkeypatch):
    cfg = read_config(get_project_root(), verbose=False, data="tpg")
    monkeypatch.setattr(cfg.data, "survey", "unknown")

    with pytest.raises(KeyError, match="unknown"):
        survey_loader(cfg.data)
