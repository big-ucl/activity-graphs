"""Shared pytest fixtures."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from activitygraphs.config import VALID_CONFIGS, Config, read_config
from activitygraphs.utils import get_project_root


@pytest.fixture(scope="session")
def load_or_skip() -> Callable:
    """Return a loader that builds an object from a dataset's config, skipping the test when its data is absent.

    Returns:
        A function ``load(data, build)`` that calls ``build(cfg, project_root)`` with the loaded
        config for ``data`` and returns its result, or skips the test if a data file is missing.
    """

    def load[T](data: VALID_CONFIGS, build: Callable[[Config, Path], T]) -> T:
        root = get_project_root()
        cfg = read_config(root, verbose=False, data=data)
        try:
            return build(cfg, root)
        except FileNotFoundError as exc:
            pytest.skip(f"{data} data not on disk: {exc}")

    return load


@pytest.fixture(scope="session")
def compose_config() -> Callable[..., Config]:
    """Return a function composing the training config with Hydra overrides, as the command line would.

    Returns:
        A function ``compose_config(*overrides)`` returning the composed ``Config``.
    """
    config_dir = str((get_project_root() / "activitygraphs/conf").resolve())

    def compose_with(*overrides: str) -> Config:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            return cast(Config, compose(config_name="config", overrides=list(overrides)))

    return compose_with
