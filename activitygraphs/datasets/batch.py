"""Typed batches: the rows of a batch of users, their join with the location graph, and the graph copies of a GNN."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from activitygraphs.datasets.locations import Locations


class UserRows(NamedTuple):
    """The tensors of one user, or of a collated batch of users.

    Wrapper to allow typing from the collated batch of users returned by ``torch.Dataset``

    Attributes:
        user_idx: Dataset index of the user, ``[]``, or ``[num_users]`` collated.
        home_node: Home node of the user, ``[]``, or ``[num_users]`` collated.
        visited: Visit labels of every node, ``[num_nodes]``, or ``[num_users, num_nodes]`` collated.
        attributes: Numeric user attributes, ``[num_numeric_attributes]``, or with a leading ``num_users`` collated.
        attributes_cat: Codes of the categorical user attributes, int64, ``[num_categorical_attributes]``, or with a
            leading ``num_users`` collated.
    """

    user_idx: Tensor
    home_node: Tensor
    visited: Tensor
    attributes: Tensor
    attributes_cat: Tensor


@dataclass(frozen=True)
class UserBatch:
    """Batch of users, one row per user, enriched with location information (e.g visited, is_home, distances ...).

    Attributes:
        user_idx: Dataset index of each user, ``[num_users]``.
        home_node: Home node of each user, ``[num_users]``.
        visited: Visit labels of every node, ``[num_users, num_nodes]``.
        is_home: One-hot indicators of whether each node is the user's home node, bool, ``[num_users, num_nodes]``.
        home_distance: Distance from the user's home to each node, as fed to the model, ``[num_users, num_nodes]``.
        home_distance_m: Distance in metres from the user's home to each node, ``[num_users, num_nodes]``.
        home_hops: Hop distance from the user's home to each node, ``[num_users, num_nodes]``.
        attributes: Numeric user attributes, ``[num_users, num_numeric_attributes]``.
        attributes_cat: Codes of the categorical user attributes, int64, ``[num_users, num_categorical_attributes]``.
    """

    user_idx: Tensor
    home_node: Tensor
    visited: Tensor
    is_home: Tensor
    home_distance: Tensor
    home_distance_m: Tensor
    home_hops: Tensor
    attributes: Tensor
    attributes_cat: Tensor

    @classmethod
    def join(cls, rows: UserRows, locations: Locations) -> "UserBatch":
        """Combine collated ``rows`` of a batch of ``UserDataset`` with the ``locations`` to return the corresp. batch.

        Args:
            rows: The collated user rows, with the same fields as ``UserDataset`` .
            locations: The location graph, on the same device as ``rows``.
        """
        home_node = rows.home_node
        is_home = F.one_hot(home_node, locations.num_nodes).bool()
        home_distance = locations.distance[home_node].to(locations.x.dtype)

        return cls(
            user_idx=rows.user_idx,
            home_node=home_node,
            visited=rows.visited,
            is_home=is_home,
            home_distance=home_distance,
            home_distance_m=locations.distance_m[home_node],
            home_hops=locations.hop_distance[home_node],
            attributes=rows.attributes,
            attributes_cat=rows.attributes_cat,
        )

    @property
    def num_users(self) -> int:
        """Number of users in the batch."""
        return self.user_idx.shape[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes of the graph."""
        return self.visited.shape[1]


@dataclass(frozen=True)
class GraphBatch:
    """The model input of a batch of users on one copy of the location graph per user, joined into one disjoint graph.

    Attributes:
        x: Model input of each node of each copy, ``[num_users * num_nodes, num_features]``.
        edge_index: Edge list, ``[2, num_users * num_edges]``.
        edge_attr: Edge features, ``[num_users * num_edges, num_edge_features]``.
        batch: Copy (user) of each node, ``[num_users * num_nodes]``.
    """

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor

    @classmethod
    def expand(cls, x: Tensor, locations: Locations) -> "GraphBatch":
        """Return the copies of the graph of ``locations`` that carry the model input ``x``, one copy per user.

        Args:
            x: Model input of every node for every user, ``[num_users, num_nodes, num_features]``.
            locations: The location graph, on the device of ``x``.

        Returns:
            The copies of the graph. Copy ``i`` holds the nodes ``i * N`` to ``(i + 1) * N - 1``, in node order, with
            ``N`` the number of nodes.
        """
        num_users, num_nodes, num_features = x.shape

        users = torch.arange(num_users, device=x.device)
        offsets = users * num_nodes
        edge_index = (locations.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)

        return cls(
            x=x.reshape(-1, num_features),
            edge_index=edge_index,
            edge_attr=locations.edge_attr.repeat(num_users, 1),
            batch=users.repeat_interleave(num_nodes),
        )
