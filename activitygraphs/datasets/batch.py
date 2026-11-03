"""Typed views over a PyG batch: the rows of a batch of users, and the copies of the location graph built from them."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from activitygraphs.datasets.locations import Locations


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


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
    """Batch of users with one copy of the location graph per user of a batch, joined into one disjoint graph.

    Attributes:
        x_loc: Location features of each node, ``[num_nodes, num_location_features]``.
        x_user_node: Per-user node features of each node, ``[num_nodes, num_user_node_features]``.
        edge_index: Edge list, ``[2, num_edges]``.
        edge_attr: Edge features, ``[num_edges, num_edge_features]``.
        y: Visit labels, ``[num_nodes, 1]``.
        batch: Graph index of each node, ``[num_nodes]``.
        ptr: Offset of each graph's first node, ``[num_graphs + 1]``.
        user_attr: Per-user numeric attributes, ``[num_graphs, num_numeric_attributes]``.
        user_attr_cat: Per-user codes of the categorical attributes, int64, ``[num_graphs, num_categorical_attributes]``.
        user_id: Dataset index of each user, ``[num_graphs]``.
        distances: Distance from the user's home to each node, as fed to the model, ``[num_nodes, 1]``.
        distances_m: Distance in metres from the user's home to each node, ``[num_nodes, 1]``.
        random_walk_pe: Random-walk positional encoding of each node, ``[num_nodes, walk_length]``.
    """

    x_loc: Tensor
    x_user_node: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    batch: Tensor
    ptr: Tensor
    user_attr: Tensor
    user_attr_cat: Tensor
    user_id: Tensor
    distances: Tensor
    distances_m: Tensor
    random_walk_pe: Tensor

    @classmethod
    def expand(cls, batch: UserBatch, locations: Locations) -> "GraphBatch":
        """Return a corresponding ``GraphBatch`` from a ``batch`` of users and ``locations``.

        Args:
            locations: The location graph, on the device of ``batch``.
            batch: The users, one copy each.

        Returns:
            The copies of the graph. Copy ``i`` holds the nodes ``i * N`` to ``(i + 1) * N - 1``, in node order, with
            ``N`` the number of nodes.
        """
        num_users = batch.num_users
        num_nodes = locations.num_nodes
        device = batch.home_node.device

        users = torch.arange(num_users, device=device)
        offsets = users * num_nodes
        edge_index = (locations.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)
        x_user_node = batch.is_home.to(locations.x.dtype).reshape(-1, 1)
        ptr = torch.arange(num_users + 1, device=device) * num_nodes

        return cls(
            x_loc=locations.x.repeat(num_users, 1),
            x_user_node=x_user_node,
            edge_index=edge_index,
            edge_attr=locations.edge_attr.repeat(num_users, 1),
            y=batch.visited.reshape(-1, 1),
            batch=users.repeat_interleave(num_nodes),
            ptr=ptr,
            user_attr=batch.attributes,
            user_attr_cat=batch.attributes_cat,
            user_id=batch.user_idx,
            distances=batch.home_distance.reshape(-1, 1),
            distances_m=batch.home_distance_m.reshape(-1, 1),
            random_walk_pe=locations.random_walk_pe.repeat(num_users, 1),
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes over all graphs in the batch."""
        return self.x_loc.shape[0]

    @property
    def num_graphs(self) -> int:
        """Number of graphs (users) in the batch."""
        return self.user_id.shape[0]
