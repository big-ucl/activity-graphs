"""[Stages 5-6: tensors -> dataset -> batches] This package handles feeding the bundled tensors into Lightning.

It is split into the following modules:
  - ``locations``: ``Locations``, the shared location graph of a survey.
  - ``users``: ``Users``, the per-user tensors of a survey.
  - ``dataset``: ``VisitDataset``, the PyG dataset combining the graph and the users.
  - ``splits``: the split strategies assigning users to train, validation and test, and the split cache.
  - ``transforms``: ``FeatureTransforms``, fitted on the training split.
  - ``datamodule``: ``prepare_splits``, ``read_splits`` and ``VisitDataModule``, the Lightning entry point.
  - ``batch``: ``UserBatch`` and ``UserGridBatch``, the typed views of a collated batch.
  - ``user_grid``: ``UserGridDataset`` and ``UserGridDataModule``, the per-user rows of a split, without the graph.
"""

from activitygraphs.datasets.batch import UserBatch, UserGridBatch
from activitygraphs.datasets.datamodule import VisitDataModule, prepare_splits, read_splits
from activitygraphs.datasets.dataset import VisitDataset, is_home_node_mask
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.splits import HouseholdSplit, SplitStrategy
from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.datasets.user_grid import UserGridDataModule, UserGridDataset
from activitygraphs.datasets.users import Users

__all__ = [
    "VisitDataModule",
    "VisitDataset",
    "FeatureTransforms",
    "Locations",
    "HouseholdSplit",
    "SplitStrategy",
    "UserBatch",
    "UserGridBatch",
    "UserGridDataModule",
    "UserGridDataset",
    "Users",
    "is_home_node_mask",
    "prepare_splits",
    "read_splits",
]
