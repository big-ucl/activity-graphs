"""VisitDataModule: LightningDataModule wrapping the prepare_splits / read_splits pipeline."""

from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T

from activitygraphs.config import Config
from activitygraphs.datasets.dataset import Fingerprint, VisitDataset, load_dataset, read_fingerprint
from activitygraphs.datasets.scalers import FittedScalers, apply_scalers, fit_scalers
from activitygraphs.datasets.splits import SplitStrategy, load_split_indices, read_split_indices
from activitygraphs.utils import get_project_root

RWPE_WALK_LENGTH = 20


def positional_encodings() -> T.Compose:
    """Return the pre-transform adding the positional encodings to the shared graph, as a separate node features."""
    return T.Compose([
        T.AddRandomWalkPE(walk_length=RWPE_WALK_LENGTH),
        # T.AddLaplacianEigenvectorPE(k=8, attr_name=None), # Removed due to instability on different graphs.
    ])


def split_cache_paths(cfg: Config, split: SplitStrategy, seed: int, project_root: Path) -> tuple[Path, Path]:
    """Return the paths of the cached split indices (``splits_*.json``) and scalers (``scalers_*.pkl``)."""
    tensors_dir = project_root / cfg.data.paths.tensors

    return tensors_dir / f"splits_{seed}_{split.key}.json", tensors_dir / f"scalers_{seed}_{split.key}.pkl"


def prepare_splits(cfg: Config, split: SplitStrategy, seed: int, project_root: Path | None = None) -> None:
    """Build and cache the dataset tensors, split indices and training-split scalers that are missing or stale."""
    project_root = get_project_root(project_root)
    splits_cache, scalers_cache = split_cache_paths(cfg, split, seed, project_root)

    dataset = None
    try:
        fingerprint = read_fingerprint(cfg, project_root)
    except FileNotFoundError:
        dataset = load_dataset(cfg, project_root=project_root, pre_transform=positional_encodings())
        fingerprint = read_fingerprint(cfg, project_root)

    if _caches_match(splits_cache, scalers_cache, fingerprint):
        return

    if dataset is None:
        dataset = load_dataset(cfg, project_root=project_root, pre_transform=positional_encodings())

    train_idx, _, _ = load_split_indices(dataset, split, seed, cache_path=splits_cache, fingerprint=fingerprint)
    is_home_idx = dataset.schema.user_nodes.index_of("is_home")

    scalers = fit_scalers(dataset, train_idx, exclude_user_node_cols=[is_home_idx], fingerprint=fingerprint)
    scalers.save(scalers_cache)


def _caches_match(splits_cache: Path, scalers_cache: Path, fingerprint: Fingerprint) -> bool:
    """Return whether the split indices and scalers are both cached for the dataset ``fingerprint``."""
    if not (splits_cache.exists() and scalers_cache.exists()):
        return False

    try:
        read_split_indices(splits_cache, fingerprint)
    except ValueError:
        return False

    return FittedScalers.read(scalers_cache).fingerprint == fingerprint


def read_splits(
    cfg: Config, split: SplitStrategy, seed: int, project_root: Path | None = None
) -> tuple[VisitDataset, VisitDataset, VisitDataset, FittedScalers]:
    """Read the dataset, split indices and scalers cached by ``prepare_splits``, and return the scaled splits.

    Returns:
        Tuple ``(train_dataset, val_dataset, test_dataset, scalers)``.

    Raises:
        FileNotFoundError: The split indices or scalers are not cached.
        ValueError: The dataset tensors are not cached, or the cached split indices or scalers were built for another
            dataset.
    """
    project_root = get_project_root(project_root)
    splits_cache, scalers_cache = split_cache_paths(cfg, split, seed, project_root)

    dataset = VisitDataset(root=str(project_root / cfg.data.paths.tensors), pre_transform=positional_encodings())
    fingerprint = read_fingerprint(cfg, project_root)

    train_idx, val_idx, test_idx = read_split_indices(splits_cache, fingerprint)

    scalers = FittedScalers.read(scalers_cache)
    if scalers.fingerprint != fingerprint:
        raise ValueError(f"Cached scalers at `{scalers_cache.name}` do not match the dataset fingerprint.")

    apply_scalers(dataset, scalers)

    return (
        cast(VisitDataset, dataset[train_idx]),
        cast(VisitDataset, dataset[val_idx]),
        cast(VisitDataset, dataset[test_idx]),
        scalers,
    )


def _compute_home_coverage(train_dataset: VisitDataset, test_dataset: VisitDataset) -> pl.DataFrame:
    """Compute each _test_ user's survey ``user_id``, home node and whether any training user also has a home there."""
    is_home_idx = train_dataset.schema.user_nodes.index_of("is_home")

    # Index of the home node of every user in the survey: [n_users].
    home_node_idx = train_dataset.user_node_features[:, :, is_home_idx].argmax(dim=1)

    train_idx = torch.as_tensor(list(train_dataset.indices()), dtype=torch.long)
    test_idx = torch.as_tensor(list(test_dataset.indices()), dtype=torch.long)
    train_home_idx = home_node_idx[train_idx].unique().tolist()

    return pl.DataFrame({
        "user_id": test_idx.tolist(),
        "survey_user_id": train_dataset.user_ids["user_id"].gather(test_idx.numpy()),
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
        self._scalers: FittedScalers | None = None
        self._home_coverage: pl.DataFrame | None = None

    def prepare_data(self) -> None:
        """Build and cache the dataset tensors, split indices and scalers that are missing or stale."""
        prepare_splits(self.cfg, self.split, self.seed, self.project_root)

    def setup(self, stage: str | None = None) -> None:
        """Read the splits and scalers cached by ``prepare_data`` and compute the home coverage, once."""
        if self._train_dataset is not None:
            return

        self._train_dataset, self._val_dataset, self._test_dataset, self._scalers = read_splits(
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
    def scalers(self) -> FittedScalers:
        """The scalers fitted on the training split."""
        if self._scalers is None:
            raise RuntimeError("Call setup() before accessing scalers.")
        return self._scalers

    @property
    def home_coverage(self) -> pl.DataFrame:
        """Per-home training-user counts, used to stratify results by home coverage."""
        if self._home_coverage is None:
            raise RuntimeError("Call setup() before accessing home_coverage.")
        return self._home_coverage
