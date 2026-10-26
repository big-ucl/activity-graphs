"""[Stages 5-6: tensors -> dataset -> batches] This package handles feeding the bundled tensors into Lightning.

It is split into the following modules:
  - ``dataset``: ``VisitDataset``, the cached PyG dataset over the survey tensors, and its fingerprint.
  - ``splits``: the split strategies assigning users to train, validation and test, and the split cache.
  - ``scalers``: ``FittedScalers``, fitted on the training split and applied in place.
  - ``datamodule``: ``prepare_splits``, ``read_splits`` and ``VisitDataModule``, the Lightning entry point.
  - ``batch``: ``UserBatch``, the typed view of a collated batch.
"""

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.datamodule import VisitDataModule, prepare_splits, read_splits
from activitygraphs.datasets.dataset import VisitDataset, is_home_node_mask, load_dataset
from activitygraphs.datasets.scalers import FittedScalers
from activitygraphs.datasets.splits import HouseholdSplit, SplitStrategy

__all__ = [
    "VisitDataModule",
    "VisitDataset",
    "FittedScalers",
    "HouseholdSplit",
    "SplitStrategy",
    "UserBatch",
    "is_home_node_mask",
    "load_dataset",
    "prepare_splits",
    "read_splits",
]
