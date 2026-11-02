"""[Stages 5-6: tensors -> dataset -> batches] This package handles feeding the bundled tensors into Lightning.

It is split into the following modules:
  - ``dataset``: ``VisitDataset``, the cached PyG dataset over the survey tensors, and its fingerprint.
  - ``splits``: the split strategies assigning users to train, validation and test, and the split cache.
  - ``transforms``: ``FeatureTransforms``, fitted on the training split and applied in place.
  - ``datamodule``: ``prepare_splits``, ``read_splits`` and ``VisitDataModule``, the Lightning entry point.
  - ``batch``: ``UserBatch`` and ``UserGridBatch``, the typed views of a collated batch.
  - ``user_grid``: ``UserGridDataset`` and ``UserGridDataModule``, the per-user rows of a split, without the graph.
"""

from activitygraphs.datasets.batch import UserBatch, UserGridBatch
from activitygraphs.datasets.datamodule import VisitDataModule, prepare_splits, read_splits
from activitygraphs.datasets.dataset import VisitDataset, is_home_node_mask, load_dataset
from activitygraphs.datasets.splits import HouseholdSplit, SplitStrategy
from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.datasets.user_grid import UserGridDataModule, UserGridDataset

__all__ = [
    "VisitDataModule",
    "VisitDataset",
    "FeatureTransforms",
    "HouseholdSplit",
    "SplitStrategy",
    "UserBatch",
    "UserGridBatch",
    "UserGridDataModule",
    "UserGridDataset",
    "is_home_node_mask",
    "load_dataset",
    "prepare_splits",
    "read_splits",
]
