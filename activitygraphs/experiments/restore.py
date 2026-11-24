"""Variants restored from the checkpoints of a recorded run, and the datamodules that score them on this run's users."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import polars as pl
from omegaconf import OmegaConf

from activitygraphs.config import VALID_SCALING_SOURCES, Config, TransferConfig
from activitygraphs.datasets import HouseholdSplit, Locations, SurveyDataModule
from activitygraphs.datasets.datamodule import read_transforms
from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.experiments.variants import Variant, build_variants, variant_hyperparams
from activitygraphs.models.architectures import build_scorer, has_node_weights
from activitygraphs.models.scorers import Scorer
from activitygraphs.runs import CHECKPOINTS_DIR, CONFIG_FILE, dataset_of_run, run_dir
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors import read_node_ids, read_schema, tensors_path
from activitygraphs.training.lightning_module import load_scorer_weights
from activitygraphs.utils import get_project_root

BEST_CHECKPOINT = "best.ckpt"


@dataclass(frozen=True)
class SourceRun:
    """A recorded run whose checkpoints, config and dataset are read to restore its variants.

    Attributes:
        path: The run directory, named by the run id.
        cfg: The config used by the run.
        variants: The variants trained by the run, keyed by name.
    """

    path: Path
    cfg: Config
    variants: dict[str, Variant]

    @classmethod
    def read(cls, runs_path: Path, run: str) -> Self:
        """Read the recorded run of id ``run`` under ``runs_path``."""
        dataset_name = dataset_of_run(run)
        path = run_dir(runs_path, dataset_name, run)
        cfg = cast(Config, OmegaConf.load(path / CONFIG_FILE))
        models = dict(cfg.models)
        trained_variants = build_variants(cfg.experiment.variants, models)
        variants_by_name = {variant.name: variant for variant in trained_variants}

        return cls(path, cfg, variants_by_name)

    @property
    def run(self) -> str:
        """The id of the run."""
        return self.path.name

    @property
    def dataset_name(self) -> str:
        """The name of the dataset of the run, such as ``TTS2016``."""
        return self.cfg.data.name

    @property
    def split(self) -> HouseholdSplit:
        """The split strategy used by the run."""
        return HouseholdSplit(val_size=self.cfg.train.val_size, test_size=self.cfg.train.test_size)

    def variant(self, name: str) -> Variant:
        """Return the variant trained by the run under the name ``name``, with the model config it was trained with.

        Raises:
            KeyError: The run trained no variant named ``name``.
        """
        if name not in self.variants:
            raise KeyError(f"Run `{self.run}` trained no variant `{name}`; it trained {sorted(self.variants)}.")

        return self.variants[name]

    def checkpoint_path(self, name: str, seed: int) -> Path:
        """Return the path of the best checkpoint of the variant ``name`` for the training ``seed``.

        Raises:
            ValueError: The run did not train with ``seed``.
            FileNotFoundError: The checkpoint is missing.
        """
        train_seeds = list(self.cfg.train.train_seeds)
        if seed not in train_seeds:
            raise ValueError(f"Run `{self.run}` trained with the seeds {train_seeds}, not with {seed}.")

        checkpoint_dir = f"{name}-seed{seed}" if len(train_seeds) > 1 else name
        path = self.path / CHECKPOINTS_DIR / checkpoint_dir / BEST_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint of `{name}` for seed {seed} at {path}.")

        return path

    def transforms(self) -> FeatureTransforms:
        """Read the feature transforms fitted on the training split of the run's dataset."""
        split_seed = self.cfg.train.split_seed

        return read_transforms(self.cfg, self.split, split_seed)

    def node_ids(self) -> pl.DataFrame:
        """Read the survey ``loc_id`` of each node of the run's dataset, one row per node."""
        project_root = get_project_root()
        tensors_root = tensors_path(self.cfg.data, project_root)

        return read_node_ids(tensors_root)

    def schema(self) -> DatasetSchema:
        """Read the column schema of each tensor group of the run's dataset."""
        project_root = get_project_root()
        tensors_root = tensors_path(self.cfg.data, project_root)

        return read_schema(tensors_root)

    def datamodule(self, batch_size: int, num_workers: int) -> SurveyDataModule:
        """Return the datamodule of the run's dataset, set up on the split used by the run."""
        split_seed = self.cfg.train.split_seed
        datamodule = SurveyDataModule(
            self.cfg,
            self.split,
            seed=split_seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.setup()

        return datamodule

    def scorer(self, name: str, seed: int, locations: Locations, schema: DatasetSchema) -> Scorer:
        """Return the scorer of the variant ``name`` over ``locations``, with the weights of its best checkpoint for ``seed``.

        The scorer is built with the model config, the input features and the home-anchored PE settings used by the
        run. Every weight of the scorer must be in the checkpoint, at the shape of the checkpoint.

        Raises:
            KeyError: The run trained no variant named ``name``.
            RuntimeError: The checkpoint does not hold exactly the weights of the scorer.
        """
        variant = self.variant(name)
        home_pe = self.cfg.train.home_pe
        scorer = build_scorer(variant.model, variant.features, locations, schema, home_pe)
        checkpoint_path = self.checkpoint_path(name, seed)
        load_scorer_weights(scorer, checkpoint_path)

        return scorer


@dataclass(frozen=True)
class RestoredVariant:
    """A variant of a recorded run, restored from its checkpoints and then tested, or fine-tuned, in this run.

    Attributes:
        source: The recorded run.
        variant: The variant as trained by the recorded run.
        scaling: The feature transforms applied to the inputs, one of ``VALID_SCALING_SOURCES``.
        fine_tune: Train the restored variant on the training split of this run before testing it. False only tests
            it.
        across_datasets: Whether the recorded run is of a dataset other than the dataset of this run.
    """

    source: SourceRun
    variant: Variant
    scaling: str
    fine_tune: bool
    across_datasets: bool

    @property
    def name(self) -> str:
        """Name of the restored variant, reported in program outputs.

        The name is the name of ``variant``, then ``from{dataset}`` when the recorded run is of another dataset, then
        ``tgtscale`` when the inputs are scaled by the transforms of this run's dataset, then ``ft`` when the variant
        is fine-tuned.
        """
        parts = [self.variant.name]
        if self.across_datasets:
            parts.append(f"from{self.source.dataset_name}")
            if self.scaling == "target":
                parts.append("tgtscale")
        if self.fine_tune:
            parts.append("ft")

        return "-".join(parts)

    @property
    def hyperparams(self) -> dict:
        """The architecture config, the input flags and the restore settings of the variant, as logged to W&B."""
        restore = {"source_run": self.source.run, "scaling": self.scaling, "fine_tune": self.fine_tune}

        return {**variant_hyperparams(self.variant), "restore": restore}


def read_source_run(runs_path: Path, transfer: TransferConfig) -> SourceRun | None:
    """Read the recorded run of ``transfer.source_run``, or return None when it is null.

    Raises:
        ValueError: ``transfer`` lists variants to restore, or fits the baselines on the recorded run's dataset,
            with no recorded run.
    """
    if transfer.source_run is not None:
        return SourceRun.read(runs_path, transfer.source_run)

    if transfer.variants or transfer.fit_baselines_on_source:
        raise ValueError("transfer.variants and transfer.fit_baselines_on_source read the run of transfer.source_run.")

    return None


def build_restored_variants(transfer: TransferConfig, source: SourceRun, dataset_name: str) -> list[RestoredVariant]:
    """Expand the variants listed by ``transfer`` into one restored variant per variant, scaling and use.

    Args:
        transfer: The transfer settings of the experiment, ``cfg.transfer``.
        source: The recorded run of ``transfer.source_run``.
        dataset_name: Name of the dataset of the experiment, ``cfg.data.name``.

    Returns:
        The restored variants, in the order of ``transfer.variants``. For each variant, the tested copies come first,
        one per scaling. The fine-tuned copies follow, one per scaling.

    Raises:
        KeyError: The recorded run trained no variant of a name in ``transfer.variants``.
        ValueError: A scaling is not one of ``VALID_SCALING_SOURCES``, a recorded run of this dataset is given a scaling other
            than ``source``, or the restored variants are neither tested nor fine-tuned.
    """
    scalings = list(transfer.scalings)
    unknown_scalings = [scaling for scaling in scalings if scaling not in VALID_SCALING_SOURCES]
    if unknown_scalings:
        raise ValueError(f"A restored variant is scaled by one of {VALID_SCALING_SOURCES}, got {unknown_scalings}.")

    across_datasets = source.dataset_name != dataset_name
    if not across_datasets and scalings != ["source"]:
        raise ValueError(
            f"Run `{source.run}` is of {dataset_name}, so its variants are already scaled by the transforms of "
            f"{dataset_name} and take the `source` scaling alone, got {scalings}."
        )

    fine_tune_options = []
    if transfer.test_restored:
        fine_tune_options.append(False)
    if transfer.fine_tune:
        fine_tune_options.append(True)
    if not fine_tune_options:
        raise ValueError(
            "The restored variants are tested, fine-tuned or both: set transfer.test_restored or fine_tune."
        )

    restored_variants = []
    for name in transfer.variants:
        variant = source.variant(name)
        for fine_tune in fine_tune_options:
            for scaling in scalings:
                restored_variants.append(RestoredVariant(source, variant, scaling, fine_tune, across_datasets))

    return restored_variants


def check_checkpoints(restored_variants: list[RestoredVariant], train_seeds: list[int]) -> None:
    """Check that the recorded run holds a checkpoint of every restored variant for every training seed.

    Raises:
        ValueError: The recorded run did not train with a seed of ``train_seeds``.
        FileNotFoundError: A checkpoint is missing.
    """
    for restored_variant in restored_variants:
        source = restored_variant.source
        name = restored_variant.variant.name
        for seed in train_seeds:
            source.checkpoint_path(name, seed)


def check_same_split(source: SourceRun, split: HouseholdSplit, split_seed: int) -> None:
    """Check that the recorded run drew its split with ``split`` and ``split_seed``, as this run does.

    Raises:
        ValueError: The recorded run drew another split, so its variants would be tested on users they were trained
            on.
    """
    source_split_seed = source.cfg.train.split_seed
    if source.split != split or source_split_seed != split_seed:
        raise ValueError(
            f"Run `{source.run}` split the users by {source.split} with seed {source_split_seed}, not by {split} with "
            f"seed {split_seed}, so its variants would be tested on users they were trained on."
        )


def check_node_weights(restored_variants: list[RestoredVariant], locations: Locations) -> None:
    """Check that every restored variant with one weight per node was trained on the nodes of ``locations``, in order.

    Raises:
        ValueError: A restored variant learned one weight per node of another dataset, whose nodes differ from the
            nodes of ``locations`` or come in another order.
    """
    for restored_variant in restored_variants:
        model = restored_variant.variant.model
        if not restored_variant.across_datasets or not has_node_weights(model):
            continue

        source = restored_variant.source
        source_node_ids = source.node_ids()
        if not source_node_ids.equals(locations.node_ids):
            raise ValueError(
                f"`{restored_variant.name}` learned one weight per node of {source.dataset_name}, whose nodes differ "
                f"from the nodes of this run's dataset or come in another order."
            )


def scaled_datamodules(cfg: Config, source: SourceRun, datamodule: SurveyDataModule) -> dict[str, SurveyDataModule]:
    """Return the datamodule of this run's users for each scaling of ``cfg.transfer.scalings``, keyed by scaling.

    The ``target`` scaling, and the ``source`` scaling of a recorded run of this dataset, keep ``datamodule``. The
    ``source`` scaling of a recorded run of another dataset gets a datamodule of the same split and users, with the
    inputs scaled by the transforms of that dataset.

    Args:
        cfg: The config used by the experiment.
        source: The recorded run of ``cfg.transfer.source_run``.
        datamodule: The datamodule of this run, scaled by the transforms of its own dataset.

    Raises:
        ValueError: The recorded run is of another dataset, with other columns than this run's dataset.
    """
    if source.dataset_name != cfg.data.name:
        source_schema = source.schema()
        _check_same_schema(source_schema, datamodule.schema, source.dataset_name, cfg.data.name)

    datamodules = {}

    for scaling in cfg.transfer.scalings:
        if scaling == "target" or source.dataset_name == cfg.data.name:
            datamodules[scaling] = datamodule
            continue

        split = HouseholdSplit(val_size=cfg.train.val_size, test_size=cfg.train.test_size)
        loader = cfg.train.loader
        source_transforms = source.transforms()
        source_scaled_datamodule = SurveyDataModule(
            cfg,
            split,
            seed=cfg.train.split_seed,
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            transforms=source_transforms,
            train_fraction=cfg.train.train_fraction,
            test_on_all_users=cfg.transfer.test_on_all_users,
        )
        source_scaled_datamodule.setup()
        datamodules[scaling] = source_scaled_datamodule

    return datamodules


def restore_scorer(restored_variant: RestoredVariant, seed: int, locations: Locations, schema: DatasetSchema) -> Scorer:
    """Return the scorer of ``restored_variant`` over ``locations``, with the weights of its best checkpoint for ``seed``.

    Raises:
        RuntimeError: The checkpoint does not hold exactly the weights of the scorer.
    """
    source = restored_variant.source
    name = restored_variant.variant.name

    return source.scorer(name, seed, locations, schema)


def _check_same_schema(source_schema: DatasetSchema, schema: DatasetSchema, source_name: str, name: str) -> None:
    """Check that two datasets have the same columns in every tensor group.

    Raises:
        ValueError: A tensor group differs, naming the columns of each dataset in every group that differs.
    """
    differences = []
    for source_group, group in zip(source_schema.groups, schema.groups, strict=True):
        if source_group != group:
            differences.append(f"`{group.group}`: {source_group.names} in {source_name}, {group.names} in {name}")

    if differences:
        raise ValueError(f"{source_name} and {name} differ in the columns of {'; '.join(differences)}.")
