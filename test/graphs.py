"""Location graphs and users built by hand for the tests, and the batches built from them."""

import polars as pl
import torch
from torch.utils.data import DataLoader

from activitygraphs.datasets import GraphBatch, Locations, UserBatch, UserDataset, Users
from activitygraphs.datasets.locations import compute_hop_distance


def chain_edge_index(num_nodes: int) -> torch.Tensor:
    """Return the edges of the chain ``0 - 1 - ... - (num_nodes - 1)``, both directions."""
    src = torch.arange(num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def make_locations(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    random_walk_pe: torch.Tensor | None = None,
    distance_m: torch.Tensor | None = None,
) -> Locations:
    """Return the locations over ``x`` and ``edge_index``, with the hop distances computed from the edges.

    A missing ``edge_attr`` or ``random_walk_pe`` has no columns, and a missing ``distance_m`` puts every node at 0 m
    from every node. The scaled ``distance`` equals ``distance_m``.
    """
    num_nodes = x.shape[0]
    if edge_attr is None:
        edge_attr = torch.zeros(edge_index.shape[1], 0)
    if random_walk_pe is None:
        random_walk_pe = torch.zeros(num_nodes, 0)
    if distance_m is None:
        distance_m = torch.zeros(num_nodes, num_nodes)

    return Locations(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        random_walk_pe=random_walk_pe,
        hop_distance=compute_hop_distance(edge_index, num_nodes),
        distance_m=distance_m,
        distance=distance_m,
        node_ids=pl.DataFrame({"loc_id": [f"z{node}" for node in range(num_nodes)]}),
    )


def make_users(
    visited: torch.Tensor,
    home_node: torch.Tensor,
    attributes: torch.Tensor | None = None,
    attributes_cat: torch.Tensor | None = None,
) -> Users:
    """Return users with the given visit labels and home nodes, each user in a household of their own.

    Missing ``attributes`` or ``attributes_cat`` have no columns.
    """
    num_users = visited.shape[0]
    if attributes is None:
        attributes = torch.zeros(num_users, 0)
    if attributes_cat is None:
        attributes_cat = torch.zeros(num_users, 0, dtype=torch.long)

    return Users(
        home_node=home_node,
        visited=visited,
        attributes=attributes,
        attributes_cat=attributes_cat,
        group_ids=torch.arange(num_users),
        user_ids=pl.DataFrame({"user_id": [f"u{user}" for user in range(num_users)]}),
    )


def loader_of(users: Users, batch_size: int) -> DataLoader:
    """Return a loader over the rows of every user of ``users``, in order."""
    indices = list(range(users.num_users))
    return DataLoader(UserDataset(users, indices), batch_size=batch_size)


def join_users(locations: Locations, users: Users) -> UserBatch:
    """Return every user of ``users`` in one batch, joined with ``locations`` as ``RankingModule`` joins its batches."""
    rows = next(iter(loader_of(users, batch_size=users.num_users)))
    return UserBatch.join(rows, locations)


def expand_users(locations: Locations, users: Users) -> GraphBatch:
    """Return one copy of the graph of ``locations`` per user of ``users``, as ``RankingModule`` builds it."""
    return GraphBatch.expand(join_users(locations, users), locations)
