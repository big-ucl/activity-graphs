"""Hydra/OmegaConf dataclass config tree and ``load_config`` helper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ===================================================
# Small sub-inputs
# ===================================================


@dataclass
class GTFSInputs:
    """File names for a GTFS feed directory."""

    directory: str

    stops: str
    stop_times: str
    trips: str
    routes: str
    agency: str
    calendar: str
    calendar_dates: str
    transfers: str


@dataclass
class OvertureInputs:
    """File names for Overture Maps land-use and place data."""

    directory: str
    land_use: str
    place: str


@dataclass
class GenevaBoundaryInputs:
    """File names for Geneva boundary shapefiles (subsectors, postcodes, boundaries)."""

    directory: str

    geneva_subsectors: str
    french_postcodes: str
    swiss_postcodes: str
    swiss_localities: str
    swiss_boundaries: str


@dataclass
class TorontoBoundaryInputs:
    """File names for Toronto boundary shapefiles (CMA, census tracts, dissemination areas)."""

    directory: str

    metropolitan_areas: str
    census_tracts: str
    dissemination_areas: str


@dataclass
class StatsInputs:
    """Base class for census statistics file config."""

    directory: str


@dataclass
class GenevaStatsInputs(StatsInputs):
    """Census statistics file config for Geneva (single grid file)."""

    file: str


@dataclass
class TorontoStatsInputs(StatsInputs):
    """Census statistics file config for Toronto (separate population and jobs CSVs)."""

    population: str
    jobs: str


# ===================================================
# Dataset-specific input
# ===================================================


@dataclass
class Inputs:
    """Base class for dataset raw-file inputs."""

    raw_journeys: str
    overture: OvertureInputs
    statistics: StatsInputs


@dataclass
class LTDSInputs(Inputs):
    """Raw-file inputs for the LTDS survey."""

    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str


@dataclass
class GenevaInputs(Inputs):
    """Raw-file inputs for the Geneva MTMC survey."""

    boundaries: GenevaBoundaryInputs
    gtfs: GTFSInputs


@dataclass
class TorontoInputs(Inputs):
    """Raw-file inputs for the Toronto TTS survey."""

    boundaries: TorontoBoundaryInputs

    raw_person: str
    raw_household: str
    raw_activities: str


# ===================================================
# Paths for datasets
# ===================================================


@dataclass
class DataPaths:
    """Filesystem paths for a dataset (raw, processed, external, PyG)."""

    processed: Path
    raw: Path

    external: Path
    gtfs: Path

    pyg_datasets: Path


# ===================================================
# Data configs
# ===================================================


@dataclass
class DataConfig:
    """Base dataset configuration: name, inputs, and paths."""

    name: str
    inputs: Inputs
    paths: DataPaths


@dataclass
class LTDSDataConfig(DataConfig):
    """Dataset config for the LTDS survey."""

    inputs: LTDSInputs


@dataclass
class GenevaDataConfig(DataConfig):
    """Dataset config for the Geneva MTMC survey."""

    inputs: GenevaInputs


@dataclass
class TorontoDataConfig(DataConfig):
    """Dataset config for the Toronto TTS survey."""

    inputs: TorontoInputs


# ===================================================
# Overall config (outputs + main)
# ===================================================


@dataclass
class OutputPaths:
    """Filesystem paths for experiment outputs (reports, figures, saved models)."""

    reports: Path
    figures: Path
    models: Path


@dataclass
class TrainConfig:
    """Training configuration for development and debugging."""

    epochs: int

    fast_dev_run: bool
    overfit_batches: int
    schedule_lr: bool
    debug: bool

    wandb: bool
    wandb_project: str
    wandb_entity: str | None


@dataclass
class Config:
    """Top-level Hydra config: dataset config plus output paths."""

    data: DataConfig
    paths: OutputPaths
    train: TrainConfig


# cs = ConfigStore.instance()
# cs.store(name="ltds_config", node=Config)


def load_config(project_root: Path, verbose=True, data: Literal["ltds", "geneva", "toronto"] = "geneva") -> Config:
    """Load Hydra config for the given dataset.

    Args:
        project_root: Repo root; the config dir is resolved as ``project_root/activitygraphs/conf``.
        verbose: Print the resolved YAML to stdout.
        data: Dataset name, one of ``ltds``, ``geneva``, or ``toronto``.

    Returns:
        Populated OmegaConf ``Config`` object.
    """
    _config_dir = str(project_root / "activitygraphs/conf")

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cfg
