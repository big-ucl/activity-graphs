"""This module contains VisitDataset, the PyG view over the locations and the users, and the split/transform cache fingerprints."""

import hashlib
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from omegaconf import OmegaConf

from activitygraphs.config import Config
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.datasets.users import Users
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors import Fingerprint, LocationTensors, UserTensors, read_tensors_fingerprint


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


class VisitDataset(pyg.data.Dataset):
    """Main PyG dataset for a travel survey: the shared ``Locations`` and the ``Users``.

    Each item returned by ``get(i)`` is a ``pyg.data.Data`` over the shared graph with additional tensors:
      - ``x_loc = locations.x``,
      - ``x_user_node``, the one-hot indicators of user ``i``'s home node,
      - ``y = users.visited[i]``,
      - ``user_attr = users.attributes[i]``, the numeric user attributes,
      - ``user_attr_cat = users.attributes_cat[i]``, the codes of the categorical user attributes,
      - ``distances = locations.distance[home]``, the model feature, scaled once the transforms are applied,
      - ``distances_m = locations.distance_m[home]``, the same distances in metres, never scaled, and
      - ``random_walk_pe = locations.random_walk_pe``.
    """

    def __init__(self, locations: Locations, users: Users, schema: DatasetSchema):
        """Wrap ``locations`` and ``users`` as a PyG dataset, one item per user.

        Args:
            locations: The shared location graph.
            users: Every user of the survey.
            schema: The column schema of each tensor group.
        """
        super().__init__()

        self.locations = locations
        self.users = users
        self.schema = schema

        self.num_users: int = users.num_users
        self.num_nodes: int = locations.num_nodes

    @classmethod
    def from_tensors(
        cls,
        locations: LocationTensors,
        users: UserTensors,
        schema: DatasetSchema,
        transforms: FeatureTransforms | None,
    ) -> Self:
        """Return the dataset over the tensors, with the features scaled by ``transforms``.

        Args:
            locations: The location tensors.
            users: The user tensors.
            schema: The column schema of each tensor group.
            transforms: The transforms fitted on the training split. ``None`` leaves the features unscaled.
        """
        return cls(Locations.from_tensors(locations, transforms), Users.from_tensors(users, transforms), schema)

    @property
    def median_realised_size(self) -> int:
        """Median number of visited nodes per user over the dataset, counting the home node."""
        return self.users.subset(list(self.indices())).median_realised_size

    def len(self) -> int:
        """Return the number of users."""
        return self.num_users

    def get(self, idx: int) -> pyg.data.Data:
        """Return user ``idx``'s graph: shared location features, and user's node features and labels."""
        if not 0 <= idx < self.num_users:
            raise IndexError(f"Index {idx} out of range for {self.num_users} users.")

        locations = self.locations
        users = self.users
        home = users.home_node[idx]
        user_node_x = F.one_hot(home, self.num_nodes).to(locations.x.dtype).unsqueeze(-1)
        y = users.visited[idx].unsqueeze(-1)

        data = pyg.data.Data(
            x_loc=locations.x,
            x_user_node=user_node_x,
            edge_index=locations.edge_index,
            edge_attr=locations.edge_attr,
            y=y,
            num_nodes=self.num_nodes,
        )

        # Unsqueeze so batching works
        data.user_attr = users.attributes[idx].unsqueeze(0)
        data.user_attr_cat = users.attributes_cat[idx].unsqueeze(0)
        data.user_id = torch.tensor([idx], dtype=torch.long)

        data.distances = locations.distance[home].to(locations.x.dtype).unsqueeze(-1)
        data.distances_m = locations.distance_m[home].unsqueeze(-1)
        data.random_walk_pe = locations.random_walk_pe

        return data


# =========================================
# Fingerprints
# =========================================


def dataset_fingerprint(cfg: Config, tensors_fp: Fingerprint) -> Fingerprint:
    """Return the fingerprint the split and transform caches are checked against: ``tensors_fp`` and the data config.

    Only ``cfg.data`` is hashed, so training hyperparameters do not invalidate the caches.
    """
    config_str = OmegaConf.to_yaml(cfg.data, resolve=True, sort_keys=True)

    return tensors_fp | {"data_config": hashlib.blake2b(config_str.encode(), digest_size=16).hexdigest()}


def read_fingerprint(cfg: Config, project_root: Path) -> Fingerprint:
    """Return the ``dataset_fingerprint`` of the cached tensors of ``cfg``, without loading them.

    Raises:
        FileNotFoundError: Any of the tensor files is not cached.
    """
    tensors_fp = read_tensors_fingerprint(project_root / cfg.data.paths.tensors)

    return dataset_fingerprint(cfg, tensors_fp)
