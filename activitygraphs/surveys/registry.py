"""The Survey loader registry,  maps the ``survey`` key of the data config to the corresp. survey loader."""

from pathlib import Path
from typing import Any

from activitygraphs.config import DataConfig
from activitygraphs.surveys import cmap, thats, tpg
from activitygraphs.surveys.base import SurveyData, SurveyLoader

REGISTRY: dict[str, SurveyLoader[Any, Any]] = {
    "tpg": tpg.LOADER,
    "thats": thats.LOADER,
    "cmap": cmap.LOADER,
}


def survey_loader(cfg: DataConfig) -> SurveyLoader[Any, Any]:
    """Return the loader corresponding to ``cfg.survey`` from the ``REGISTRY``.

    Raises:
        KeyError: If no loader is registered in ``cfg.survey``.
    """
    if cfg.survey not in REGISTRY:
        raise KeyError(f"No survey loader registered under {cfg.survey!r}, expected one of {sorted(REGISTRY)}")

    return REGISTRY[cfg.survey]


def load_survey(cfg: DataConfig, project_root: Path | None = None) -> SurveyData:
    """Load the survey in ``cfg.survey`` from the ``processed`` cache, builds the survey if miss."""
    return survey_loader(cfg).load(cfg, project_root)
