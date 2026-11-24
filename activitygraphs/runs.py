"""This module names, creates and looks up the run directories. Each directory contains everything written by one experiment.

A run is named by its id ``{dataset}-{experiment}-{YYYYmmdd-HHMMSS}``, which is also the W&B group of its models. A run
is stored in ``{id}/`` under ``paths.runs``, and an overfit run under ``paths.overfit``.
"""

import re
from datetime import datetime
from pathlib import Path

RESULTS_FILE = "results.parquet"
PER_USER_FILE = "per-user.parquet"
SCORES_FILE = "scores.parquet"
DISTANCES_FILE = "distances.parquet"
FITTED_PARAMS_FILE = "fitted-params.parquet"
CONFIG_FILE = "config.yaml"
HYDRA_LINK = "hydra"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_ID_PATTERN = re.compile(r"(?P<dataset>[A-Za-z0-9]+)-(?P<experiment>[A-Za-z0-9_]+)-(?P<started>\d{8}-\d{6})")


def run_id(name: str, experiment: str, started: datetime) -> str:
    """Return the id of the run of ``experiment`` on the dataset ``name`` started at ``started``.

    Raises:
        ValueError: The dataset or the experiment name holds a character other than a letter, a digit or, in the
            experiment name, an underscore.
    """
    run = f"{name}-{experiment}-{started:{TIMESTAMP_FORMAT}}"
    if RUN_ID_PATTERN.fullmatch(run) is None:
        raise ValueError(f"`{run}` is not a run id: dataset and experiment names hold letters, digits and underscores.")

    return run


def dataset_of_run(run: str) -> str:
    """Return the name of the dataset of the run ``run``.

    Raises:
        ValueError: ``run`` is not a run id.
    """
    match = RUN_ID_PATTERN.fullmatch(run)
    if match is None:
        raise ValueError(f"`{run}` is not a run id `{{dataset}}-{{experiment}}-{{YYYYmmdd-HHMMSS}}`.")

    return match.group("dataset")


def create_run_dir(root: Path, run: str) -> Path:
    """Create the directory of the new run ``run`` under ``root``.

    Raises:
        FileExistsError: The run already has a directory.
    """
    path = root / run
    path.mkdir(parents=True)

    return path


def run_dir(runs_path: Path, name: str, run: str | None = None) -> Path:
    """Resolve the directory of the run ``run`` of the dataset ``name``, or of its latest run with results if None.

    Raises:
        ValueError: ``run`` is not a run id of the dataset ``name``.
        FileNotFoundError: No directory has that id.
    """
    run = _latest_run(runs_path, name) if run is None else run
    if dataset_of_run(run) != name:
        raise ValueError(f"Run `{run}` is not a run of {name}.")

    path = runs_path / run
    if not path.is_dir():
        raise FileNotFoundError(f"no run {run} in {runs_path}")

    return path


def latest_overfit_dir(overfit_path: Path, name: str) -> Path:
    """Find the directory of the most recently started overfit run of ``name``.

    Raises:
        FileNotFoundError: No overfit run of ``name`` exists.
    """
    paths = _run_dirs(overfit_path, name)
    if not paths:
        raise FileNotFoundError(f"no {name} overfit run in {overfit_path}")

    return paths[-1]


def _latest_run(runs_path: Path, name: str) -> str:
    """Return the id of the latest-started run of the dataset ``name`` whose directory holds a results file.

    Raises:
        FileNotFoundError: No run of ``name`` has results.
    """
    finished_run_dirs = [path for path in _run_dirs(runs_path, name) if (path / RESULTS_FILE).exists()]
    if not finished_run_dirs:
        raise FileNotFoundError(f"no {name} run with a {RESULTS_FILE} in {runs_path}")

    return finished_run_dirs[-1].name


def _run_dirs(root: Path, name: str) -> list[Path]:
    """List every run directory of the dataset ``name`` under ``root``, with or without results, in start order."""
    if not root.is_dir():
        return []

    runs = []
    for path in root.iterdir():
        match = RUN_ID_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None and match.group("dataset") == name:
            runs.append((match.group("started"), path))

    return [path for _, path in sorted(runs)]
