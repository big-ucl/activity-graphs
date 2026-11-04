"""This module contains UserDataset, the users of one split, and the split/transform cache fingerprints."""

import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from activitygraphs.config import Config
from activitygraphs.datasets.batch import UserRows
from activitygraphs.datasets.users import Users
from activitygraphs.tensors import Fingerprint, read_tensors_fingerprint, tensors_path


class UserDataset(torch.utils.data.Dataset):
    """The users of one split, one item per user.

    Each item holds the ``UserRows`` of one user.

    Args:
        users: Every user of the survey.
        indices: Dataset indices of the users of the split.

    Attributes:
        users: The users of the split, in the order of ``indices``.
        user_idx: Dataset index of each user of the split, ``[num_split_users]``.
    """

    def __init__(self, users: Users, indices: list[int]):
        self.users = users.subset(indices)
        self.user_idx = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of users of the split."""
        return len(self.user_idx)

    def __getitem__(self, index: int) -> UserRows:
        """Return the rows of the ``index``-th user of the split."""
        users = self.users

        return UserRows(
            user_idx=self.user_idx[index],
            home_node=users.home_node[index],
            visited=users.visited[index],
            attributes=users.attributes[index],
            attributes_cat=users.attributes_cat[index],
        )


# =========================================
# Fingerprints
# =========================================


def read_fingerprint(cfg: Config, project_root: Path) -> Fingerprint:
    """Return the ``_dataset_fingerprint`` of the cached tensors of ``cfg``, without loading them.

    Raises:
        FileNotFoundError: Any of the tensor files is not cached.
    """
    tensors_fp = read_tensors_fingerprint(tensors_path(cfg.data, project_root))

    return _dataset_fingerprint(cfg, tensors_fp)


def _dataset_fingerprint(cfg: Config, tensors_fp: Fingerprint) -> Fingerprint:
    """Return the fingerprint the split and transform caches are checked against: ``tensors_fp`` and the data config.

    Only ``cfg.data`` is hashed, so training hyperparameters do not invalidate the caches.
    """
    config_str = OmegaConf.to_yaml(cfg.data, resolve=True, sort_keys=True)

    return tensors_fp | {"data_config": hashlib.blake2b(config_str.encode(), digest_size=16).hexdigest()}
