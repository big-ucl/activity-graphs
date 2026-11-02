"""Users: the per-user tensors of a survey, one row per user."""

from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import polars as pl
import torch

from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.tensors import UserTensors


@dataclass(frozen=True)
class Users:
    """The per-user tensors of a survey, one row per user.

    Attributes:
        home_node: Home node of each user (as node indices), int64, ``[num_users]``.
        visited: Visit labels of each user over every node, ``[num_users, num_nodes]``.
        attributes: Numeric user attributes, ``[num_users, n_numeric_attr]``.
        attributes_cat: Categorical user attribute codes, int64, ``[num_users, n_categorical_attr]``.
        group_ids: Household group id of each user, ``[num_users]``.
        user_ids: Survey ``user_id`` and ``hh_id`` of each user, one row per user.
    """

    home_node: torch.Tensor
    visited: torch.Tensor
    attributes: torch.Tensor
    attributes_cat: torch.Tensor
    group_ids: torch.Tensor
    user_ids: pl.DataFrame

    @classmethod
    def from_tensors(cls, tensors: UserTensors, transforms: FeatureTransforms | None) -> Self:
        """Return the users of ``tensors``, with the attributes scaled by ``transforms``.

        Args:
            tensors: The user tensors.
            transforms: The transforms fitted on the training split. ``None`` leaves the attributes unscaled.
        """
        if transforms is None:
            attributes = tensors.attributes
        else:
            attributes = transforms.groups["user_attributes"].apply(tensors.attributes)

        return cls(
            home_node=tensors.home_node,
            visited=tensors.visited,
            attributes=attributes,
            attributes_cat=tensors.attributes_cat,
            group_ids=tensors.group_ids,
            user_ids=tensors.user_ids,
        )

    @property
    def num_users(self) -> int:
        """Number of users."""
        return self.user_ids.height

    @property
    def median_realised_size(self) -> int:
        """Median number of visited nodes per user, counting the home node."""
        sizes = self.visited.sum(dim=1).numpy()
        return int(np.median(sizes))

    def subset(self, indices: list[int]) -> Self:
        """Return the users at ``indices``, in that order."""
        return replace(
            self,
            home_node=self.home_node[indices],
            visited=self.visited[indices],
            attributes=self.attributes[indices],
            attributes_cat=self.attributes_cat[indices],
            group_ids=self.group_ids[indices],
            user_ids=self.user_ids[indices],
        )
