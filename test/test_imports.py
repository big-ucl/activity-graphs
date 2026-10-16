"""Smoke test that every module in the package imports."""

import importlib
import pkgutil

import pytest

import activitygraphs

MODULES = [m.name for m in pkgutil.walk_packages(activitygraphs.__path__, prefix="activitygraphs.")]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    importlib.import_module(module)
