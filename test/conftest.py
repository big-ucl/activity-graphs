"""Shared pytest fixtures."""

from collections.abc import Callable
from pathlib import Path

import pytest

from activitygraphs.config import VALID_CONFIGS, Config, load_config
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
        cfg = load_config(root, verbose=False, data=data)
        try:
            return build(cfg, root)
        except FileNotFoundError as exc:
            pytest.skip(f"{data} data not on disk: {exc}")

    return load
