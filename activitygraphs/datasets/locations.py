"""Locations: the shared location graph of a survey, its node features, edges and node-to-node distances."""

from dataclasses import dataclass, fields, replace
from typing import Self

import numpy as np
import polars as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from activitygraphs.datasets.transforms import FeatureTransforms
from activitygraphs.schema import EdgeKind
from activitygraphs.tensors import EdgeTensors, LocationTensors

RWPE_WALK_LENGTH = 20


@dataclass(frozen=True)
class Locations:
    """The shared location graph of a survey, held once for every user.

    Attributes:
        x: Location features, ``[num_nodes, num_location_features]``.
        access: Accessibility features of the locations, ``[num_nodes, num_access_features]``.
        edges: Edges between the nodes with their scaled features, for every ``EdgeKind``.
        random_walk_pe: Random-walk positional encoding of each node, ``[num_nodes, num_rwpe_features]``.
        hop_distance: Graph-hop distance between every pair of nodes over the contiguity edges, ``inf`` for unreachable pairs,
            ``[num_nodes, num_nodes]``.
        network_hop_distance: Graph-hop distance between every pair of nodes over the contiguity and rail edges,
            ``inf`` for unreachable pairs, ``[num_nodes, num_nodes]``.
        distance_m: Distance between the centroids of every pair of nodes in metres, never scaled,
            ``[num_nodes, num_nodes]``. Row ``i`` holds the distances from node ``i`` to every node.
        distance: Scaled and transformed ``distance_m`, used as a model feature, ``[num_nodes, num_nodes]``.
        travel_times_min: Travel time in minutes and reachability by each mode from every node to every node, never
            scaled, ``[num_nodes, num_nodes, num_travel_time_features]``. Row ``i`` holds the travel times from node
            ``i``.
        travel_times: Scaled and transformed ``travel_times_min``, used as a model feature,
            ``[num_nodes, num_nodes, num_travel_time_features]``.
        node_ids: Survey ``loc_id`` of each node, one row per node.
    """

    x: torch.Tensor
    access: torch.Tensor
    edges: dict[EdgeKind, EdgeTensors]
    random_walk_pe: torch.Tensor
    hop_distance: torch.Tensor
    network_hop_distance: torch.Tensor
    distance_m: torch.Tensor
    distance: torch.Tensor
    travel_times_min: torch.Tensor
    travel_times: torch.Tensor
    node_ids: pl.DataFrame

    @classmethod
    def from_tensors(cls, tensors: LocationTensors, transforms: FeatureTransforms | None) -> Self:
        """Return the locations of ``tensors``, with the features scaled by ``transforms``.

        Args:
            tensors: The location tensors.
            transforms: The transforms fitted on the training split. ``None`` leaves the features unscaled.
        """
        contiguity_edge_index = tensors.edges[EdgeKind.CONTIGUITY].index
        graph = pyg.data.Data(edge_index=contiguity_edge_index, num_nodes=tensors.num_nodes)
        random_walk_pe = T.AddRandomWalkPE(walk_length=RWPE_WALK_LENGTH)(graph).random_walk_pe
        hop_distance = _compute_hop_distance(contiguity_edge_index, tensors.num_nodes)

        if transforms is None:
            x, access, edges, distance = tensors.x, tensors.access, tensors.edges, tensors.distance_m
            travel_times = tensors.travel_times_min
        else:
            distance_m = tensors.distance_m.unsqueeze(-1)
            x = transforms.groups["locations"].apply(tensors.x)
            access = transforms.groups["access"].apply(tensors.access)
            edges = {}
            for kind, kind_edges in tensors.edges.items():
                scaled_attr = transforms.groups[kind.group].apply(kind_edges.attr)
                edges[kind] = EdgeTensors(kind_edges.index, scaled_attr)
            distance = transforms.groups["distances"].apply(distance_m).squeeze(-1)
            travel_times = transforms.groups["travel_times"].apply(tensors.travel_times_min)

        network_edge_index = torch.cat([kind_edges.index for kind_edges in edges.values()], dim=1)
        network_hop_distance = _compute_hop_distance(network_edge_index, tensors.num_nodes)

        return cls(
            x=x,
            access=access,
            edges=edges,
            random_walk_pe=random_walk_pe,
            hop_distance=hop_distance,
            network_hop_distance=network_hop_distance,
            distance_m=tensors.distance_m,
            distance=distance,
            travel_times_min=tensors.travel_times_min,
            travel_times=travel_times,
            node_ids=tensors.node_ids,
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes of the graph."""
        return self.node_ids.height

    @property
    def num_rwpe_features(self) -> int:
        """Number of columns of the random-walk positional encoding, ``random_walk_pe``."""
        return self.random_walk_pe.shape[1]

    def to(self, device: torch.device) -> Self:
        """Return the locations with every tensor on ``device``."""
        moved: dict[str, torch.Tensor | dict[EdgeKind, EdgeTensors]] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                moved[field.name] = value.to(device)
            elif isinstance(value, dict):
                moved[field.name] = {kind: edges.to(device) for kind, edges in value.items()}

        return replace(self, **moved)


def _compute_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the all-pairs hop distance matrix over ``edge_index``, ``[num_nodes, num_nodes]``, float32.

    Treats the graph as undirected and unweighted; unreachable pairs are ``inf``.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
    hop_distance = shortest_path(adjacency, method="D", unweighted=True, directed=False)

    return torch.as_tensor(hop_distance, dtype=torch.float)
