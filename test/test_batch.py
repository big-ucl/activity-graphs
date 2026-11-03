"""Unit tests for the rows of a batch of users, their join with the location graph, and the graph copies."""

from dataclasses import replace

import torch
from graphs import make_locations, make_users
from torch.utils.data import DataLoader

from activitygraphs.datasets import GraphBatch, UserBatch, UserDataset

DISTANCE_M = torch.tensor([[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]])


def make_path_locations():
    """Return the path ``0 - 1 - 2`` with one feature per node and per edge and two encoding columns per node."""
    locations = make_locations(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
        random_walk_pe=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        distance_m=DISTANCE_M,
    )
    distance = DISTANCE_M / 10.0
    return replace(locations, distance=distance)


def make_two_users():
    """Return eight users, of whom user 7 lives at node 2 and user 4 at node 0, and each of the two visits two nodes."""
    visited = torch.zeros(8, 3)
    visited[7] = torch.tensor([1.0, 1.0, 0.0])
    visited[4] = torch.tensor([1.0, 0.0, 1.0])
    home_node = torch.zeros(8, dtype=torch.long)
    home_node[7] = 2
    attributes = torch.arange(8.0).unsqueeze(-1)
    attributes_cat = torch.arange(8).unsqueeze(-1)

    return make_users(visited, home_node, attributes, attributes_cat)


def join_two_users() -> tuple[UserBatch, GraphBatch]:
    locations = make_path_locations()
    rows = next(iter(DataLoader(UserDataset(make_two_users(), [7, 4]), batch_size=2)))
    batch = UserBatch.join(rows, locations)
    return batch, GraphBatch.expand(batch, locations)


def test_an_item_holds_the_rows_of_one_user():
    item = UserDataset(make_two_users(), [7, 4])[1]

    assert item.user_idx == 4
    assert item.home_node == 0
    assert item.visited.tolist() == [1.0, 0.0, 1.0]
    assert item.attributes.tolist() == [4.0]
    assert item.attributes_cat.tolist() == [4]


def test_the_split_holds_its_users_in_the_order_of_the_indices():
    split = UserDataset(make_two_users(), [7, 4])

    assert split.user_idx.tolist() == [7, 4]
    assert split.users.home_node.tolist() == [2, 0]
    assert split.users.user_ids["user_id"].to_list() == ["u7", "u4"]


def test_the_join_reads_each_users_row_of_the_location_graph():
    batch, _ = join_two_users()

    assert batch.user_idx.tolist() == [7, 4]
    assert batch.is_home.tolist() == [[False, False, True], [True, False, False]]
    assert batch.home_distance_m.tolist() == [[9.0, 4.0, 0.0], [0.0, 5.0, 9.0]]
    torch.testing.assert_close(batch.home_distance, torch.tensor([[0.9, 0.4, 0.0], [0.0, 0.5, 0.9]]))
    assert batch.home_hops.tolist() == [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]
    assert batch.visited.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]


def test_the_expansion_holds_one_copy_of_the_graph_per_user():
    _, graph = join_two_users()

    assert graph.x_loc.squeeze(-1).tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    assert graph.x_user_node.squeeze(-1).tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert graph.edge_index.tolist() == [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]]
    assert graph.edge_attr.squeeze(-1).tolist() == [10.0, 11.0, 12.0, 13.0] * 2
    assert graph.y.squeeze(-1).tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert graph.batch.tolist() == [0, 0, 0, 1, 1, 1]
    assert graph.ptr.tolist() == [0, 3, 6]
    assert graph.user_id.tolist() == [7, 4]
    assert graph.user_attr.tolist() == [[7.0], [4.0]]
    assert graph.user_attr_cat.tolist() == [[7], [4]]
    torch.testing.assert_close(graph.distances.squeeze(-1), torch.tensor([0.9, 0.4, 0.0, 0.0, 0.5, 0.9]))
    assert graph.distances_m.squeeze(-1).tolist() == [9.0, 4.0, 0.0, 0.0, 5.0, 9.0]
    torch.testing.assert_close(graph.random_walk_pe, make_path_locations().random_walk_pe.repeat(2, 1))
    assert graph.num_nodes == 6
    assert graph.num_graphs == 2


def test_moving_the_locations_moves_every_tensor_and_keeps_the_node_ids():
    locations = make_path_locations().to(torch.device("meta"))

    assert locations.x.is_meta
    assert locations.edge_index.is_meta
    assert locations.edge_attr.is_meta
    assert locations.random_walk_pe.is_meta
    assert locations.hop_distance.is_meta
    assert locations.distance_m.is_meta
    assert locations.distance.is_meta
    assert locations.node_ids["loc_id"].to_list() == ["z0", "z1", "z2"]
