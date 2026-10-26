"""Split strategies, each assigning every user of a dataset to the train, validation or test split, and the split cache."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.model_selection import GroupShuffleSplit

from activitygraphs.datasets.dataset import Fingerprint, VisitDataset

type SplitIndices = tuple[list[int], list[int], list[int]]


class SplitStrategy(Protocol):
    """A rule assigning every user of a dataset to the train, validation or test split."""

    @property
    def key(self) -> str:
        """Name of the strategy and its parameters in the split and scaler cache file names."""
        ...

    def split(self, dataset: VisitDataset, seed: int) -> SplitIndices:
        """Return the ``(train, val, test)`` user indices of ``dataset`` drawn with ``seed``."""
        ...


@dataclass(frozen=True)
class HouseholdSplit:
    """Random split by household: all the users of one household fall in the same split.

    Attributes:
        val_size: Fraction of users reserved for validation.
        test_size: Fraction of users reserved for testing.
    """

    val_size: float
    test_size: float

    @property
    def key(self) -> str:
        """Cache key naming the split fractions."""
        return f"val{self.val_size}_test{self.test_size}_household"

    def split(self, dataset: VisitDataset, seed: int) -> SplitIndices:
        """Return the ``(train, val, test)`` user indices of ``dataset``, grouped by ``dataset.group_ids``."""
        all_idx = list(range(len(dataset)))

        # Split Test and Train-Val
        gss_test = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=seed)
        train_val_idx, test_idx = next(iter(gss_test.split(all_idx, groups=dataset.group_ids)))

        # Compute size of validation set and adjust the group Ids
        train_val_group_ids = dataset.group_ids[train_val_idx]
        proportional_val_size = self.val_size / (1 - self.test_size)

        # Split Val and Train
        gss_val = GroupShuffleSplit(n_splits=1, test_size=proportional_val_size, random_state=seed)
        rel_train_idx, rel_val_idx = next(iter(gss_val.split(train_val_idx, groups=train_val_group_ids)))
        train_idx, val_idx = train_val_idx[rel_train_idx], train_val_idx[rel_val_idx]

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def read_split_indices(cache_path: Path, fingerprint: Fingerprint) -> SplitIndices:
    """Return the (train_indices, val_indices, test_indices) cached by ``load_split_indices`` at ``cache_path``.

    Raises:
        FileNotFoundError: Nothing is cached at ``cache_path``.
        ValueError: The cached indices were split from a dataset other than ``fingerprint``.
    """
    with cache_path.open() as f:
        cached_indices = json.load(f)

    if cached_indices["fingerprint"] != fingerprint:
        raise ValueError(f"Cached indices at `{cache_path.name}` do not match the dataset fingerprint.")

    return cached_indices["train"], cached_indices["val"], cached_indices["test"]


def load_split_indices(
    dataset: VisitDataset,
    split: SplitStrategy,
    seed: int,
    fingerprint: Fingerprint,
    cache_path: Path | None = None,
) -> SplitIndices:
    """Return (train_indices, val_indices, test_indices). Loads from ``cache_path`` if it matches ``fingerprint``, otherwise splits and caches."""
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            cached_indices = json.load(f)

        if cached_indices.get("fingerprint") == fingerprint:
            return cached_indices["train"], cached_indices["val"], cached_indices["test"]

        print(f"Cached indices at `{cache_path.name}` do not match fingerprint, re-splitting.")

    train_idx, val_idx, test_idx = split.split(dataset, seed)

    assert set(train_idx) & set(val_idx) == set() and set(train_idx + val_idx) & set(test_idx) == set(), (
        "Assert train/val/test sets are disjoint."
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(
                {"train": train_idx, "val": val_idx, "test": test_idx, "seed": seed, "fingerprint": fingerprint}, f
            )

    return train_idx, val_idx, test_idx
