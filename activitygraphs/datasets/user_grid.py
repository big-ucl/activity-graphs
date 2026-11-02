"""UserGridDataset and UserGridDataModule: the per-user rows of a ``VisitDataset`` split, without the shared graph."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from activitygraphs.datasets.datamodule import VisitDataModule
from activitygraphs.datasets.dataset import VisitDataset


class UserGridDataset(Dataset):
    """The per-user tensors of a ``VisitDataset`` split, one item per user, without the shared graph.

    Each item is a dict with the keys of ``UserGridBatch``: the user's dataset index ``user_id``, home node
    ``home_node``, the visit labels ``y`` and the distances ``distances`` and ``distances_m`` of every node, each
    ``[num_nodes]``, and the attributes ``user_attr`` and ``user_attr_cat``.

    Args:
        split: A split of a ``VisitDataset``, after the transforms are applied.
    """

    def __init__(self, split: VisitDataset):
        self.split = split
        self.user_idx = torch.as_tensor(list(split.indices()), dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of users of the split."""
        return len(self.user_idx)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the per-user tensors of the ``index``-th user of the split."""
        user = self.user_idx[index]
        locations = self.split.locations
        users = self.split.users
        home = users.home_node[user]

        return {
            "user_id": user,
            "home_node": home,
            "y": users.visited[user],
            "distances": locations.distance[home].float(),
            "distances_m": locations.distance_m[home],
            "user_attr": users.attributes[user],
            "user_attr_cat": users.attributes_cat[user],
        }


class UserGridDataModule(L.LightningDataModule):
    """LightningDataModule over the splits of a ``VisitDataModule``, whose loaders yield ``UserGridDataset`` batches.

    Args:
        visits: The ``VisitDataModule`` that builds and reads the splits and transforms.
    """

    def __init__(self, visits: VisitDataModule):
        super().__init__()
        self.visits = visits

    def prepare_data(self) -> None:
        """Build and cache the dataset tensors, split indices and transforms that are missing or stale."""
        self.visits.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        """Read the splits and transforms cached by ``prepare_data``, once."""
        self.visits.setup(stage)

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled loader over the users of the training split."""
        users = UserGridDataset(self.visits.train_dataset)
        return DataLoader(users, batch_size=self.visits.batch_size, shuffle=True, num_workers=self.visits.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return a loader over the users of the validation split."""
        users = UserGridDataset(self.visits.val_dataset)
        return DataLoader(users, batch_size=self.visits.batch_size, num_workers=self.visits.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return a loader over the users of the test split."""
        users = UserGridDataset(self.visits.test_dataset)
        return DataLoader(users, batch_size=self.visits.batch_size, num_workers=self.visits.num_workers)
