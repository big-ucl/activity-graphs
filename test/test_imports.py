"""Smoke test that every module in the package imports, and the import direction between its packages."""

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import activitygraphs

MODULES = [m.name for m in pkgutil.walk_packages(activitygraphs.__path__, prefix="activitygraphs.")]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    importlib.import_module(module)


def imported_modules(path: Path) -> set[str]:
    """Return the names of the modules imported by the source file at ``path``."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


MODELS_SOURCES = sorted((Path(activitygraphs.__path__[0]) / "models").rglob("*.py"))


@pytest.mark.parametrize("path", MODELS_SOURCES, ids=lambda path: path.name)
def test_models_never_imports_training(path: Path):
    training = [name for name in imported_modules(path) if name.split(".")[:2] == ["activitygraphs", "training"]]

    assert training == []
