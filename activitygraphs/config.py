"""Hydra/OmegaConf dataclass config tree and ``read_config`` helper."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ===================================================
# Small sub-inputs
# ===================================================


@dataclass
class GTFSFeeds:
    """GTFS feeds of a dataset and corresponding service date from which to read timetable data.

    Attributes:
        service_date: An ISO date, a weekday within the survey start and end dates.
        feeds: Path to each GTFS, as a directory or a zip file, keyed by feed name.
    """

    service_date: str
    feeds: dict[str, str]


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
class TTSBoundaryPaths:
    """Boundary files of the TTS traffic zones and of the census dissemination areas."""

    directory: str
    zones: str
    dissemination_areas: str


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


@dataclass
class TTSStatsPaths(StatsPaths):
    """Census statistics file config for one TTS survey version (the population of each dissemination area)."""

    population: str


# ===================================================
# Dataset-specific input
# ===================================================


@dataclass
class SurveyPaths:
    """Base class for dataset raw-file inputs."""

    raw_journeys: str
    overture: OverturePaths


@dataclass
class PTPaths(SurveyPaths):
    """Base class for the raw-file inputs of a dataset with public transport feeds and an OSM extract.

    Attributes:
        gtfs: The GTFS feeds of the dataset and assoc. service date.
        osm_extract: File name of the OSM extract clipped to the zones, in ``paths.osm``.
    """

    gtfs: GTFSFeeds
    osm_extract: str


@dataclass
class TPGPaths(PTPaths):
    """Raw file paths for the TPG survey."""

    statistics: TPGStatsPaths
    boundaries: TPGBoundaryPaths


@dataclass
class THATSPaths(SurveyPaths):
    """Raw file inputs for the Toronto THATS survey."""

    statistics: THATSStatsPaths
    boundaries: THATSBoundaryPaths

    raw_person: str
    raw_household: str
    raw_activities: str


@dataclass
class CMAPPaths(PTPaths):
    """Raw file inputs for the CMAP MyDailyTravel survey."""

    statistics: CMAPStatsPaths
    boundaries: CMAPBoundaryPaths

    raw_person: str
    raw_household: str


@dataclass
class TTSPaths(PTPaths):
    """Raw file inputs for one survey version of the Transportation Tomorrow Survey."""

    statistics: TTSStatsPaths
    boundaries: TTSBoundaryPaths

    raw_person: str
    raw_household: str


# ===================================================
# Paths for datasets
# ===================================================


@dataclass
class DataPaths:
    """Filesystem paths for a dataset (raw, processed, external)."""

    processed: Path
    raw: Path

    external: Path
    gtfs: Path
    osm: Path


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
    train_batch_fraction: float
    max_epochs: int


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


@dataclass
class TTSDataConfig(DataConfig):
    """Dataset config for a TTS survey version, 2016 or 2022."""

    inputs: TTSPaths  # pyrefly: ignore[bad-override-mutable-attribute]
    survey_version: int


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
class TrainerConfig:
    """Keyword arguments of the Lightning ``Trainer`` of every learned model."""

    max_epochs: int
    accelerator: str
    gradient_clip_val: float
    fast_dev_run: bool
    overfit_batches: int
    limit_train_batches: float
    enable_progress_bar: bool
    enable_model_summary: bool


@dataclass
class LoaderConfig:
    """DataLoader settings of every split."""

    batch_size: int
    num_workers: int


@dataclass
class LRSchedulerConfig:
    """Keyword arguments of the ``ReduceLROnPlateau`` scheduler."""

    factor: float
    patience: int


@dataclass
class EarlyStoppingConfig:
    """Early stopping on the loss's monitored validation metric.

    Attributes:
        patience: Number of epochs without improvement after which a model stops training. None trains every model
            for ``trainer.max_epochs``.
    """

    patience: int | None


@dataclass
class HomePEConfig:
    """Home-anchored positional encoding settings.

    Attributes:
        bins: Number of RBF centres, which is the number of feature columns added.
        max_hops: Hop distance at which the encoding is capped, and the position of the last centre.
    """

    bins: int
    max_hops: int


@dataclass
class TrainConfig:
    """Training configuration for development and debugging."""

    split_seed: int
    train_seeds: list[int]

    loader: LoaderConfig
    val_size: float
    test_size: float

    overfit_lr: float | None
    overfit_epochs: int
    schedule_lr: bool
    lr_scheduler: LRSchedulerConfig
    early_stopping: EarlyStoppingConfig
    home_pe: HomePEConfig
    calibration_bins: int
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
class VariantConfig:
    """Config for one variant in the variant list of an experiment.

    Attributes:
        model: Key of the architecture in ``cfg.models``.
        set: Model config fields overridden for this variant.
        features: Flags of ``InputFeatures`` overridden for this variant. A flag left out keeps its default.
        sweep: At most one model config field, mapped to the values it is trained at, one variant per value. Empty
            trains one variant.
    """

    model: str
    # Any: the value type depends on the key, and OmegaConf 2.3 rejects a union element type in a list.
    set: dict[str, Any] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Config for the experiment to run, from the ``experiment`` config group.

    Attributes:
        name: Name of the experiment, which labels its W&B runs.
        variants: The variants trained by the experiment, in training order.
    """

    name: str
    variants: list[VariantConfig]


@dataclass
class ModelConfig:
    """Config for one architecture, from the ``model`` config group. Contains name, optimiser settings and model arch.

    Attributes:
        architecture: Key of the builder in ``models.architectures.BUILDERS`` that constructs the scorer.
        name: Name the architecture is reported under; a variant appends its flags to it.
        lr: Initial AdamW learning rate.
        weight_decay: AdamW weight decay.
    """

    architecture: str
    name: str
    lr: float
    weight_decay: float


@dataclass
class NetworkConfig(ModelConfig):
    """Config for an neural network architecture.

    Attributes:
        hidden_channels: Width of the hidden layers.
        dropout: Dropout probability.
    """

    hidden_channels: int
    dropout: float


@dataclass
class MLPConfig(NetworkConfig):
    """Config for the node-wise MLP (``NodeMLP``).

    Attributes:
        num_layers: Number of linear layers.
    """

    num_layers: int


@dataclass
class GATConfig(NetworkConfig):
    """Config for the GATv2 with skip connections (``GAT``).

    Attributes:
        num_pre_layers: Number of linear layers before message passing.
        num_gcn_layers: Number of message-passing layers.
        num_post_layers: Number of linear layers after message passing.
        residuals: Add residual connections around the message-passing layers.
        norm: Normalisation applied to each message-passing layer's input, or None for none.
    """

    num_pre_layers: int
    num_gcn_layers: int
    num_post_layers: int
    residuals: bool
    norm: Literal["layer_norm", "batch_norm"] | None


@dataclass
class GPSConfig(NetworkConfig):
    """Config for the GPS graph transformer (``GraphTransformer``).

    Attributes:
        num_layers: Number of GPS layers.
        num_heads: Number of attention heads per layer.
    """

    num_layers: int
    num_heads: int


@dataclass
class TwoTowerConfig(GATConfig):
    """Config for the two-tower scorer (``TwoTowerScorer``), whose node tower is a ``GAT`` with the fields of ``GATConfig``.

    Attributes:
        embedding_dim: Width ``d`` of the node and user embeddings.
        num_user_layers: Number of linear layers of the user tower.
        num_pair_layers: Number of linear layers of the pairwise block.
        bias: The bias of each node. ``location`` is a linear function of the node embedding. ``embedding_mlp`` is an
            MLP of the node embedding. ``feature_mlp`` is an MLP of the location features, through a city-specific
            projection to width ``d``. ``node`` is a free parameter per node, an in-city diagnostic.
        num_bias_layers: Number of linear layers of the ``embedding_mlp`` and ``feature_mlp`` biases.
        pair_location_features: Give the pairwise block the location features, through a city-specific projection
            to width ``d``.
        node_embedding: Append a learned embedding of each node's identity, of width ``d``, to the node tower's
            input. It is an in-city diagnostic.
    """

    embedding_dim: int
    num_user_layers: int
    num_pair_layers: int
    bias: Literal["location", "embedding_mlp", "feature_mlp", "node"]
    num_bias_layers: int
    pair_location_features: bool
    node_embedding: bool


@dataclass
class HomeZoneMFConfig(ModelConfig):
    """Config for the home-zone matrix factorisation (``HomeZoneMFBaseline``).

    Attributes:
        n_factors: Latent dimension of the home and node factors.
    """

    n_factors: int


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
    trainer: TrainerConfig
    train: TrainConfig
    experiment: ExperimentConfig
    models: dict[str, ModelConfig]
    analysis: AnalysisConfig


VALID_CONFIGS = Literal["tpg", "thats", "cmap", "tts2016", "tts2022"]


def read_config(project_root: Path, verbose=True, data: VALID_CONFIGS = "tpg") -> Config:
    """Read the Hydra config for the given dataset.

    Args:
        project_root: Repo root; the config dir is resolved as ``project_root/activitygraphs/conf``.
        verbose: Print the resolved YAML to stdout.
        data: Dataset name, one of ``cmap``, ``thats``, ``tpg``, ``tts2016`` or ``tts2022``.

    Returns:
        Populated OmegaConf ``Config`` object.
    """
    _config_dir = str((project_root / "activitygraphs/conf").resolve())

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cast(Config, cfg)
