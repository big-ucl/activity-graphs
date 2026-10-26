"""Tests that every composed config node has exactly the keys of the dataclass in ``config.py`` that types it."""

import dataclasses
import typing

import pytest
from omegaconf import DictConfig, OmegaConf

from activitygraphs.config import (
    CMAPDataConfig,
    Config,
    DepthSweepConfig,
    ExperimentConfig,
    GATConfig,
    GPSConfig,
    MLPConfig,
    THATSDataConfig,
    TPGDataConfig,
)
from activitygraphs.utils import get_project_root

CONF_DIR = get_project_root() / "activitygraphs/conf"

DATA_CONFIGS = {"tpg": TPGDataConfig, "cmap": CMAPDataConfig, "thats": THATSDataConfig}
MODEL_CONFIGS = {"mlp": MLPConfig, "gat": GATConfig, "gps": GPSConfig}
EXPERIMENT_CONFIGS = {"depth_sweep": DepthSweepConfig}

# Typed by the per-option dataclasses above rather than by the base classes ``Config`` declares.
OPTION_NODES = {"data", "models", "experiment"}


def mismatches(cls: type, node: DictConfig, path: str, skip: set[str] = frozenset()) -> list[str]:
    """Return the keys of ``node`` that are not fields of ``cls`` and the reverse, recursing into dataclass fields."""
    hints = typing.get_type_hints(cls)
    fields = {field.name for field in dataclasses.fields(cls)}
    keys = set(node.keys())

    found = [f"{path}.{key}: not a field of {cls.__name__}" for key in sorted(keys - fields)]
    found += [f"{path}.{key}: missing from the yaml" for key in sorted(fields - keys)]

    for key in sorted((fields & keys) - skip):
        if dataclasses.is_dataclass(hints[key]) and isinstance(node[key], DictConfig):
            found += mismatches(hints[key], node[key], f"{path}.{key}")

    return found


def test_shared_nodes_match_config(compose_config):
    assert mismatches(Config, compose_config(), "cfg", skip=OPTION_NODES) == []


@pytest.mark.parametrize("name", sorted(DATA_CONFIGS))
def test_data_config_matches_its_dataclass(compose_config, name):
    assert mismatches(DATA_CONFIGS[name], compose_config(f"data={name}").data, "data") == []


@pytest.mark.parametrize("path", sorted((CONF_DIR / "model").glob("*.yaml")), ids=lambda path: path.stem)
def test_model_config_matches_its_dataclass(compose_config, path):
    architecture = OmegaConf.load(path).architecture
    node = compose_config(f"model@models.{architecture}={path.stem}").models[architecture]

    assert mismatches(MODEL_CONFIGS[architecture], node, "model") == []


@pytest.mark.parametrize("path", sorted((CONF_DIR / "experiment").glob("*.yaml")), ids=lambda path: path.stem)
def test_experiment_config_matches_its_dataclass(compose_config, path):
    cls = EXPERIMENT_CONFIGS.get(path.stem, ExperimentConfig)

    assert mismatches(cls, compose_config(f"experiment={path.stem}").experiment, "experiment") == []
