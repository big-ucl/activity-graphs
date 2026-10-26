"""Hydra/OmegaConf dataclass config tree and ``read_config`` helper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ===================================================
# Small sub-inputs
# ===================================================


@dataclass
class GTFSPaths:
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
class OverturePaths:
    """File names for Overture Maps land-use and place data."""

    directory: str
    land_use: str
    place: str


@dataclass
class TPGBoundaryPaths:
    """File names for Geneva boundary shapefiles (subsectors, postcodes, boundaries)."""

    directory: str

    geneva_subsectors: str
    french_postcodes: str
    swiss_postcodes: str
    swiss_localities: str
    swiss_boundaries: str


@dataclass
class THATSBoundaryPaths:
    """File names for Toronto THATS boundary shapefiles (CMA, census tracts, dissemination areas)."""

    directory: str

    metropolitan_areas: str
    census_tracts: str
    dissemination_areas: str


@dataclass
class CMAPBoundaryPaths:
    """File names for CMAP MyDailyTravel boundary shapefiles (census tracts)."""

    directory: str
    census_tracts: str


@dataclass
class StatsPaths:
    """Base class for census statistics file config."""

    directory: str


@dataclass
class TPGStatsPaths(StatsPaths):
    """Census statistics file config for Geneva (square grid of job and population data)."""

    file: str


@dataclass
class THATSStatsPaths(StatsPaths):
    """Census statistics file config for THATS (separate population and jobs CSVs)."""

    population: str
    jobs: str


@dataclass
class CMAPStatsPaths(StatsPaths):
    """Census statistics file config for CMAP MyDailyTravel survey."""

    population: str
    jobs: str


# ===================================================
# Dataset-specific input
# ===================================================


@dataclass
class SurveyPaths:
    """Base class for dataset raw-file inputs."""

    raw_journeys: str
    overture: OverturePaths


@dataclass
class TPGPaths(SurveyPaths):
    """Raw file paths for the TPG survey."""

    statistics: TPGStatsPaths
    boundaries: TPGBoundaryPaths
    gtfs: GTFSPaths


@dataclass
class THATSPaths(SurveyPaths):
    """Raw file inputs for the Toronto THATS survey."""

    statistics: THATSStatsPaths
    boundaries: THATSBoundaryPaths

    raw_person: str
    raw_household: str
    raw_activities: str


@dataclass
class CMAPPaths(SurveyPaths):
    """Raw file inputs for the CMAP MyDailyTravel survey."""

    statistics: CMAPStatsPaths
    boundaries: CMAPBoundaryPaths

    raw_person: str
    raw_household: str


# ===================================================
# Paths for datasets
# ===================================================


@dataclass
class DataPaths:
    """Filesystem paths for a dataset (raw, processed, external, tensors)."""

    processed: Path
    raw: Path

    external: Path
    gtfs: Path

    tensors: Path


# ===================================================
# Data configs
# ===================================================


@dataclass
class DataConfig:
    """Base dataset configuration: name, survey loader key, inputs, paths, and the distance bands the region is split into."""

    name: str
    survey: str
    inputs: SurveyPaths
    paths: DataPaths
    distance_bands: list[float]


@dataclass
class TPGDataConfig(DataConfig):
    """Dataset config for the TPG survey."""

    inputs: TPGPaths  # pyrefly: ignore[bad-override-mutable-attribute]


@dataclass
class THATSDataConfig(DataConfig):
    """Dataset config for the Toronto THATS survey."""

    inputs: THATSPaths  # pyrefly: ignore[bad-override-mutable-attribute]


@dataclass
class CMAPDataConfig(DataConfig):
    """Dataset config for the Chicago CMAP MyDailyTravel 2022 survey."""

    inputs: CMAPPaths  # pyrefly: ignore[bad-override-mutable-attribute]


# ===================================================
# Overall config (outputs + main)
# ===================================================


@dataclass
class OutputPaths:
    """Filesystem paths for experiment outputs (reports, run directories, overfit run directories, figures)."""

    reports: Path
    runs: Path
    overfit: Path
    figures: Path


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
class ModelConfig:
    """Config for one architecture, from the ``model`` config group. Contains name, optimiser settings and model arch.

    Attributes:
        architecture: Key of the builder in ``experiments.architectures.BUILDERS`` that constructs the model.
        name: Name the architecture is reported under; a variant appends its flags to it.
        lr: Initial AdamW learning rate.
        weight_decay: AdamW weight decay.
        hidden_channels: Width of the hidden layers.
        dropout: Dropout probability.
    """

    architecture: str
    name: str
    lr: float
    weight_decay: float
    hidden_channels: int
    dropout: float


@dataclass
class MLPConfig(ModelConfig):
    """Config for the node-wise MLP (``NodeMLP``).

    Attributes:
        num_layers: Number of linear layers.
    """

    num_layers: int


@dataclass
class GATConfig(ModelConfig):
    """Config for the GAT with skip connections (``GATSkip``).

    Attributes:
        num_pre_layers: Number of linear layers before message passing.
        num_gcn_layers: Number of message-passing layers.
        num_post_layers: Number of linear layers after message passing.
        residuals: Add residual connections around the message-passing layers.
    """

    num_pre_layers: int
    num_gcn_layers: int
    num_post_layers: int
    residuals: bool


@dataclass
class GPSConfig(ModelConfig):
    """Config for the GPS graph transformer (``GraphTransformer``).

    Attributes:
        num_layers: Number of GPS layers.
        num_heads: Number of attention heads per layer.
    """

    num_layers: int
    num_heads: int


@dataclass
class AnalysisConfig:
    """Result analysis configuration."""

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
    models: dict[str, ModelConfig]
    analysis: AnalysisConfig


VALID_CONFIGS = Literal["tpg", "thats", "cmap"]


def read_config(project_root: Path, verbose=True, data: VALID_CONFIGS = "tpg") -> Config:
    """Read the Hydra config for the given dataset.

    Args:
        project_root: Repo root; the config dir is resolved as ``project_root/activitygraphs/conf``.
        verbose: Print the resolved YAML to stdout.
        data: Dataset name, one of ``cmap``, ``thats``, or ``tpg``.

    Returns:
        Populated OmegaConf ``Config`` object.
    """
    _config_dir = str((project_root / "activitygraphs/conf").resolve())

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cast(Config, cfg)
