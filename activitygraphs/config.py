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
class THATSBoundaryInputs:
    """File names for Toronto THATS boundary shapefiles (CMA, census tracts, dissemination areas)."""

    directory: str

    metropolitan_areas: str
    census_tracts: str
    dissemination_areas: str


@dataclass
class CMAPBoundaryInputs:
    """File names for CMAP MyDailyTravel boundary shapefiles (census tracts)."""

    directory: str
    census_tracts: str


@dataclass
class StatsInputs:
    """Base class for census statistics file config."""

    directory: str


@dataclass
class GenevaStatsInputs(StatsInputs):
    """Census statistics file config for Geneva (square grid of job and population data)."""

    file: str


@dataclass
class THATSStatsInputs(StatsInputs):
    """Census statistics file config for THATS (separate population and jobs CSVs)."""

    population: str
    jobs: str


@dataclass
class CMAPStatsInputs(StatsInputs):
    """Census statistics file config for CMAP MyDailyTravel survey."""

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
    """Raw file inputs for the LTDS survey."""

    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str


@dataclass
class GenevaInputs(Inputs):
    """Raw file inputs for the Geneva MTMC survey."""

    boundaries: GenevaBoundaryInputs
    gtfs: GTFSInputs


@dataclass
class THATSInputs(Inputs):
    """Raw file inputs for the Toronto THATS survey."""

    boundaries: THATSBoundaryInputs

    raw_person: str
    raw_household: str
    raw_activities: str


@dataclass
class CMAPInputs(Inputs):
    """Raw file inputs for the CMAP MyDailyTravel survey."""

    boundaries: CMAPBoundaryInputs

    raw_person: str
    raw_household: str


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
    """Base dataset configuration: name, inputs, paths, and the distance bands the region is split into."""

    name: str
    inputs: Inputs
    paths: DataPaths
    distance_bands: list[float]


@dataclass
class LTDSDataConfig(DataConfig):
    """Dataset config for the LTDS survey."""

    inputs: LTDSInputs


@dataclass
class GenevaDataConfig(DataConfig):
    """Dataset config for the Geneva MTMC survey."""

    inputs: GenevaInputs


@dataclass
class THATSDataConfig(DataConfig):
    """Dataset config for the Toronto THATS survey."""

    inputs: THATSInputs


@dataclass
class CMAPDataConfig(DataConfig):
    """Dataset config for the Chicago CMAP MyDailyTravel 2022 survey."""

    inputs: CMAPInputs


# ===================================================
# Overall config (outputs + main)
# ===================================================


@dataclass
class OutputPaths:
    """Filesystem paths for experiment outputs (reports, figures, saved models)."""

    reports: Path
    reports_data: Path
    figures: Path
    models: Path


@dataclass
class LossConfig:
    """Loss configuration: type (BPR or BCE) and associated params."""

    type: Literal["bce", "bpr"] = "bpr"
    n_pairs: int = 128
    neg_sampler: Literal["uniform", "hard"] = "uniform"
    is_near_threshold: int = 5


@dataclass
class TrainConfig:
    """Training configuration for development and debugging."""

    epochs: int
    split_seed: int
    train_seeds: list[int]

    experiment: Literal["comparison", "depth_sweep", "demographics_ablation", "overfit_health", "baselines"]
    depths: list[int]

    batch_size: int
    val_size: float
    test_size: float

    fast_dev_run: bool
    overfit_batches: int
    overfit_lr: float | None
    overfit_epochs: int
    schedule_lr: bool
    compile: bool
    debug: bool
    save_score_vectors: bool
    log_train_ranking: bool
    max_recall_k: int

    wandb: bool
    wandb_project: str
    wandb_entity: str | None

    loss: LossConfig


@dataclass
class AnalysisConfig:
    """Result analysis configuration"""

    reference_model: str
    main_metric: str
    per_user_metric: str
    min_realised_size: int
    occupancy_thresholds: list[int]
    recall_curve_ks: list[int]
    run: int | None


@dataclass
class Config:
    """Top-level Hydra config: dataset config plus output paths."""

    data: DataConfig
    paths: OutputPaths
    train: TrainConfig
    analysis: AnalysisConfig


# cs = ConfigStore.instance()
# cs.store(name="ltds_config", node=Config)

VALID_CONFIGS = Literal["ltds", "geneva", "thats", "cmap"]


def load_config(project_root: Path, verbose=True, data: VALID_CONFIGS = "geneva") -> Config:
    """Load Hydra config for the given dataset.

    Args:
        project_root: Repo root; the config dir is resolved as ``project_root/activitygraphs/conf``.
        verbose: Print the resolved YAML to stdout.
        data: Dataset name, one of ``ltds``, ``geneva``, or ``thats``.

    Returns:
        Populated OmegaConf ``Config`` object.
    """
    _config_dir = str(project_root / "activitygraphs/conf")

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cfg
