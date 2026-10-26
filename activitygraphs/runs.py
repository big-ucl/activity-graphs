"""This module creates, numbers and looks up the run directories. Each directory contains everything written by one experiment.

A run of dataset ``name`` is stored in ``{name}-{N}-{YYYYmmdd-HHMMSS}/`` under ``paths.runs``, numbered in start order; an
overfit run lives in ``{name}-{YYYYmmdd-HHMMSS}/`` under ``paths.overfit``, unnumbered.
"""

import re
from datetime import datetime
from pathlib import Path

RESULTS_FILE = "results.parquet"
PER_USER_FILE = "per-user.parquet"
SCORES_FILE = "scores.parquet"
DISTANCES_FILE = "distances.parquet"
CONFIG_FILE = "config.yaml"
HYDRA_LINK = "hydra"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def create_run_dir(runs_path: Path, name: str, started: datetime) -> Path:
    """Create the directory of a new numbered run, one above the highest number taken by any run of ``name``.

    Returns:
        The created directory, ``runs_path/{name}-{N}-{timestamp}``.
    """
    taken = numbered_run_dirs(runs_path, name)
    run = max(taken, default=0) + 1
    path = runs_path / f"{name}-{run}-{started:{TIMESTAMP_FORMAT}}"
    path.mkdir(parents=True)

    return path


def create_overfit_dir(overfit_path: Path, name: str, started: datetime) -> Path:
    """Create the directory of a new overfit run, ``overfit_path/{name}-{timestamp}``."""
    path = overfit_path / f"{name}-{started:{TIMESTAMP_FORMAT}}"
    path.mkdir(parents=True)

    return path


def numbered_run_dirs(runs_path: Path, name: str) -> dict[int, list[Path]]:
    """List every run directory of ``name`` under ``runs_path``, keyed by run number, with or without results."""
    pattern = re.compile(rf"{re.escape(name)}-(\d+)-\d{{8}}-\d{{6}}")
    runs: dict[int, list[Path]] = {}

    if not runs_path.is_dir():
        return runs

    for path in runs_path.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            runs.setdefault(int(match.group(1)), []).append(path)

    return runs


def latest_run(runs_path: Path, name: str) -> int:
    """Find the highest run number of ``name`` whose directory holds a results file.

    Raises:
        FileNotFoundError: No run of ``name`` has results.
    """
    finished = [
        run
        for run, paths in numbered_run_dirs(runs_path, name).items()
        if any((path / RESULTS_FILE).exists() for path in paths)
    ]

    if not finished:
        raise FileNotFoundError(f"no {name} run with a {RESULTS_FILE} in {runs_path}")

    return max(finished)


def run_dir(runs_path: Path, name: str, run: int | None = None) -> Path:
    """Resolve the directory of run ``run`` of ``name``, or of the latest run with results if ``run`` is None.

    Raises:
        FileNotFoundError: No directory has that run number.
        ValueError: Several directories have that run number.
    """
    run = latest_run(runs_path, name) if run is None else run
    paths = numbered_run_dirs(runs_path, name).get(run, [])

    if not paths:
        raise FileNotFoundError(f"no {name} run {run} in {runs_path}")
    if len(paths) > 1:
        raise ValueError(f"{name} run {run} matches several directories: {sorted(p.name for p in paths)}")

    return paths[0]


def latest_overfit_dir(overfit_path: Path, name: str) -> Path:
    """Find the directory of the most recently started overfit run of ``name``.

    Raises:
        FileNotFoundError: No overfit run of ``name`` exists.
    """
    pattern = re.compile(rf"{re.escape(name)}-\d{{8}}-\d{{6}}")
    paths = sorted(p for p in overfit_path.iterdir() if p.is_dir() and pattern.fullmatch(p.name))

    if not paths:
        raise FileNotFoundError(f"no {name} overfit run in {overfit_path}")

    return paths[-1]
