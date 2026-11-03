"""[Stages 5-6: tensors -> dataset -> batches] This package handles feeding the bundled tensors into Lightning.

It is split into the following modules:
  - ``locations``: ``Locations``, the shared location graph of a survey.
  - ``users``: ``Users``, the per-user tensors of a survey.
  - ``dataset``: ``UserDataset``, the users of one dataset split, and the split/transform cache fingerprints.
  - ``splits``: the split strategies assigning users to train, validation and test, and the split cache.
  - ``transforms``: ``FeatureTransforms``, fitted on the training split.
  - ``datamodule``: ``prepare_splits``, ``read_splits`` and ``SurveyDataModule``, the Lightning entry point.
  - ``batch``: ``UserRows``, the tensors of one user or of a collated batch of users, ``UserBatch``, a batch of
    users joined with the location graph, and ``GraphBatch``, the model input on one copy of the graph per user.
"""

from activitygraphs.datasets.batch import GraphBatch, UserBatch, UserRows
from activitygraphs.datasets.datamodule import SurveyDataModule, prepare_splits, read_splits
from activitygraphs.datasets.dataset import UserDataset
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.splits import HouseholdSplit, SplitStrategy
from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.datasets.users import Users

__all__ = [
    "FeatureTransforms",
    "GraphBatch",
    "HouseholdSplit",
    "Locations",
    "SplitStrategy",
    "SurveyDataModule",
    "UserBatch",
    "UserDataset",
    "UserRows",
    "Users",
    "prepare_splits",
    "read_splits",
]
