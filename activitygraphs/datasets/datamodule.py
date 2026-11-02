"""VisitDataModule: LightningDataModule wrapping the prepare_splits / read_splits pipeline."""

from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import torch
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.datasets.dataset import VisitDataset, read_fingerprint
from activitygraphs.datasets.splits import SplitStrategy, load_split_indices, read_split_indices
from activitygraphs.datasets.transforms import FeatureTransforms, fit_transforms
from activitygraphs.tensors import Fingerprint, load_tensors, read_tensors
from activitygraphs.utils import get_project_root


def split_cache_paths(cfg: Config, split: SplitStrategy, seed: int, project_root: Path) -> tuple[Path, Path]:
    """Return the paths of the cached split indices (``splits_*.json``) and transforms (``transforms_*.pkl``)."""
    tensors_dir = project_root / cfg.data.paths.tensors

    return tensors_dir / f"splits_{seed}_{split.key}.json", tensors_dir / f"transforms_{seed}_{split.key}.pkl"


def prepare_splits(cfg: Config, split: SplitStrategy, seed: int, project_root: Path | None = None) -> None:
    """Build and cache the dataset tensors, split indices and training-split transforms that are missing or stale."""
    project_root = get_project_root(project_root)
    splits_cache, transforms_cache = split_cache_paths(cfg, split, seed, project_root)

    tensors = None
    try:
        fingerprint = read_fingerprint(cfg, project_root)
    except FileNotFoundError:
        tensors = load_tensors(cfg, project_root)
        fingerprint = read_fingerprint(cfg, project_root)

    if _caches_match(splits_cache, transforms_cache, fingerprint):
        return

    if tensors is None:
        tensors = load_tensors(cfg, project_root)

    locations, users, schema = tensors
    train_idx, _, _ = load_split_indices(users, split, seed, cache_path=splits_cache, fingerprint=fingerprint)

    transforms = fit_transforms(locations, users, schema, train_idx, fingerprint=fingerprint)
    transforms.save(transforms_cache)


def _caches_match(splits_cache: Path, transforms_cache: Path, fingerprint: Fingerprint) -> bool:
    """Return whether the split indices and the current transforms are both cached for the dataset ``fingerprint``."""
    if not (splits_cache.exists() and transforms_cache.exists()):
        return False

    try:
        read_split_indices(splits_cache, fingerprint)
    except ValueError:
        return False

    return FeatureTransforms.read(transforms_cache).is_current(fingerprint)


def read_splits(
    cfg: Config, split: SplitStrategy, seed: int, project_root: Path | None = None
) -> tuple[VisitDataset, VisitDataset, VisitDataset, FeatureTransforms]:
    """Read the dataset, split indices and transforms cached by ``prepare_splits``, and return the transformed splits.

    Returns:
        Tuple ``(train_dataset, val_dataset, test_dataset, transforms)``.

    Raises:
        FileNotFoundError: The dataset tensors, split indices or transforms are not cached.
        ValueError: The cached split indices or transforms were built for another dataset, or the cached transforms
            follow another transform policy.
    """
    project_root = get_project_root(project_root)
    splits_cache, transforms_cache = split_cache_paths(cfg, split, seed, project_root)

    locations, users, schema = read_tensors(project_root / cfg.data.paths.tensors)
    fingerprint = read_fingerprint(cfg, project_root)

    train_idx, val_idx, test_idx = read_split_indices(splits_cache, fingerprint)

    transforms = FeatureTransforms.read(transforms_cache)
    if not transforms.is_current(fingerprint):
        raise ValueError(
            f"Cached transforms at `{transforms_cache.name}` do not match the dataset fingerprint or the transform policy."
        )

    dataset = VisitDataset.from_tensors(locations, users, schema, transforms)

    return (
        cast(VisitDataset, dataset[train_idx]),
        cast(VisitDataset, dataset[val_idx]),
        cast(VisitDataset, dataset[test_idx]),
        transforms,
    )


def _compute_home_coverage(train_dataset: VisitDataset, test_dataset: VisitDataset) -> pl.DataFrame:
    """Compute each _test_ user's survey ``user_id``, home node and whether any training user also has a home there."""
    home_node_idx = train_dataset.users.home_node

    train_idx = torch.as_tensor(list(train_dataset.indices()), dtype=torch.long)
    test_idx = torch.as_tensor(list(test_dataset.indices()), dtype=torch.long)
    train_home_idx = home_node_idx[train_idx].unique().tolist()

    return pl.DataFrame({
        "user_id": test_idx.tolist(),
        "survey_user_id": train_dataset.users.user_ids["user_id"].gather(test_idx.numpy()),
        "home_node": home_node_idx[test_idx].tolist(),
    }).with_columns(home_seen_in_train=pl.col("home_node").is_in(train_home_idx))


class VisitDataModule(L.LightningDataModule):
    """LightningDataModule wrapping ``prepare_splits`` and ``read_splits`` for activity graph prediction.

    ``prepare_data()`` builds the missing caches with ``prepare_splits``; ``setup()`` reads them once with
    ``read_splits``, builds PyG DataLoaders, and exposes the home coverage of the test users.

    Args:
        cfg: Hydra config containing dataset paths.
        split: Strategy assigning the users to the train, validation and test splits.
        seed: Random seed for the train/validation split.
        batch_size: Number of graphs per DataLoader batch.
        num_workers: Number of DataLoader worker processes.
        project_root: Optional override for the project root path.
    """

    def __init__(
        self,
        cfg: Config,
        split: SplitStrategy,
        seed: int,
        batch_size: int,
        num_workers: int,
        project_root: Path | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_root = project_root

        self._train_dataset: VisitDataset | None = None
        self._val_dataset: VisitDataset | None = None
        self._test_dataset: VisitDataset | None = None
        self._transforms: FeatureTransforms | None = None
        self._home_coverage: pl.DataFrame | None = None

    def prepare_data(self) -> None:
        """Build and cache the dataset tensors, split indices and transforms that are missing or stale."""
        prepare_splits(self.cfg, self.split, self.seed, self.project_root)

    def setup(self, stage: str | None = None) -> None:
        """Read the splits and transforms cached by ``prepare_data`` and compute the home coverage, once."""
        if self._train_dataset is not None:
            return

        self._train_dataset, self._val_dataset, self._test_dataset, self._transforms = read_splits(
            self.cfg, self.split, self.seed, self.project_root
        )

        assert self._train_dataset is not None
        assert self._test_dataset is not None

        self._home_coverage = _compute_home_coverage(self._train_dataset, self._test_dataset)

    def train_dataloader(self) -> pyg.loader.DataLoader:
        """Return a shuffled loader over the training split."""
        assert self._train_dataset is not None
        return pyg.loader.DataLoader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> pyg.loader.DataLoader:
        """Return a loader over the validation split."""
        assert self._val_dataset is not None
        return pyg.loader.DataLoader(self._val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> pyg.loader.DataLoader:
        """Return a loader over the test split."""
        assert self._test_dataset is not None
        return pyg.loader.DataLoader(self._test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def train_dataset(self) -> VisitDataset:
        """The training split."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> VisitDataset:
        """The validation split."""
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> VisitDataset:
        """The test split."""
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    @property
    def transforms(self) -> FeatureTransforms:
        """The transforms fitted on the training split."""
        if self._transforms is None:
            raise RuntimeError("Call setup() before accessing transforms.")
        return self._transforms

    @property
    def home_coverage(self) -> pl.DataFrame:
        """Per-home training-user counts, used to stratify results by home coverage."""
        if self._home_coverage is None:
            raise RuntimeError("Call setup() before accessing home_coverage.")
        return self._home_coverage
