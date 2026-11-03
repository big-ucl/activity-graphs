"""SurveyDataModule: LightningDataModule wrapping the prepare_splits / read_splits pipeline."""

from pathlib import Path

import lightning as L
import polars as pl
from torch.utils.data import DataLoader

from activitygraphs.config import Config
from activitygraphs.datasets.dataset import UserDataset, read_fingerprint
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.splits import SplitStrategy, load_split_indices, read_split_indices
from activitygraphs.datasets.transforms import FeatureTransforms, fit_transforms
from activitygraphs.datasets.users import Users
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors import Fingerprint, load_tensors, read_tensors, tensors_path
from activitygraphs.utils import get_project_root

DATASETS_DIR = "Datasets"


def split_cache_paths(cfg: Config, split: SplitStrategy, seed: int, project_root: Path) -> tuple[Path, Path]:
    """Return the paths of the cached split indices (``splits_*.json``) and transforms (``transforms_*.pkl``).

    Both caches are under ``cfg.data.paths.processed/Datasets/``.
    """
    datasets_dir = project_root / cfg.data.paths.processed / DATASETS_DIR

    return datasets_dir / f"splits_{seed}_{split.key}.json", datasets_dir / f"transforms_{seed}_{split.key}.pkl"


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
) -> tuple[Locations, DatasetSchema, UserDataset, UserDataset, UserDataset]:
    """Read the tensors, split indices and transforms cached by ``prepare_splits``, and return the scaled splits.

    Returns:
        Tuple ``(locations, schema, train_dataset, val_dataset, test_dataset)``.

    Raises:
        FileNotFoundError: The dataset tensors, split indices or transforms are not cached.
        ValueError: The cached split indices or transforms were built for another dataset, or the cached transforms
            follow another transform policy.
    """
    project_root = get_project_root(project_root)
    splits_cache, transforms_cache = split_cache_paths(cfg, split, seed, project_root)

    location_tensors, user_tensors, schema = read_tensors(tensors_path(cfg.data, project_root))
    fingerprint = read_fingerprint(cfg, project_root)

    train_idx, val_idx, test_idx = read_split_indices(splits_cache, fingerprint)

    transforms = FeatureTransforms.read(transforms_cache)
    if not transforms.is_current(fingerprint):
        raise ValueError(
            f"Cached transforms at `{transforms_cache.name}` do not match the dataset fingerprint or the transform policy."
        )

    locations = Locations.from_tensors(location_tensors, transforms)
    users = Users.from_tensors(user_tensors, transforms)

    return (
        locations,
        schema,
        UserDataset(users, train_idx),
        UserDataset(users, val_idx),
        UserDataset(users, test_idx),
    )


def _compute_home_coverage(train_dataset: UserDataset, test_dataset: UserDataset) -> pl.DataFrame:
    """Compute each _test_ user's survey ``user_id``, home node and whether any training user also has a home there."""
    train_home_idx = train_dataset.users.home_node.unique().tolist()

    return pl.DataFrame({
        "user_id": test_dataset.user_idx.tolist(),
        "survey_user_id": test_dataset.users.user_ids["user_id"],
        "home_node": test_dataset.users.home_node.tolist(),
    }).with_columns(home_seen_in_train=pl.col("home_node").is_in(train_home_idx))


class SurveyDataModule(L.LightningDataModule):
    """LightningDataModule of one survey: location graph and the users of each split.

    ``prepare_data()`` builds the missing caches with ``prepare_splits``. ``setup()`` reads them once with
    ``read_splits`` and exposes the home coverage of the test users. Each loader yields the rows of the users of a
    split, one ``UserDataset`` item per user. Joining the batches with the location graph is not done here, but
    by ``RankingModule``.

    Args:
        cfg: Hydra config containing dataset paths.
        split: Strategy assigning the users to the train, validation and test splits.
        seed: Random seed for the train/validation split.
        batch_size: Number of users per DataLoader batch.
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

        self._locations: Locations | None = None
        self._schema: DatasetSchema | None = None
        self._train_dataset: UserDataset | None = None
        self._val_dataset: UserDataset | None = None
        self._test_dataset: UserDataset | None = None
        self._home_coverage: pl.DataFrame | None = None

    def prepare_data(self) -> None:
        """Build and cache the dataset tensors, split indices and transforms that are missing or stale."""
        prepare_splits(self.cfg, self.split, self.seed, self.project_root)

    def setup(self, stage: str | None = None) -> None:
        """Read the splits and transforms cached by ``prepare_data`` and compute the home coverage, once."""
        if self._train_dataset is not None:
            return

        self._locations, self._schema, self._train_dataset, self._val_dataset, self._test_dataset = read_splits(
            self.cfg, self.split, self.seed, self.project_root
        )
        self._home_coverage = _compute_home_coverage(self._train_dataset, self._test_dataset)

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled loader over the users of the training split."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return a loader over the users of the validation split."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return a loader over the users of the test split."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def locations(self) -> Locations:
        """The location graph, scaled by the transforms fitted on the training split."""
        if self._locations is None:
            raise RuntimeError("Call setup() before accessing locations.")
        return self._locations

    @property
    def schema(self) -> DatasetSchema:
        """The column schema of each tensor group."""
        if self._schema is None:
            raise RuntimeError("Call setup() before accessing schema.")
        return self._schema

    @property
    def train_dataset(self) -> UserDataset:
        """The training split."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> UserDataset:
        """The validation split."""
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> UserDataset:
        """The test split."""
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    @property
    def home_coverage(self) -> pl.DataFrame:
        """Per-home training-user counts, used to stratify results by home coverage."""
        if self._home_coverage is None:
            raise RuntimeError("Call setup() before accessing home_coverage.")
        return self._home_coverage
