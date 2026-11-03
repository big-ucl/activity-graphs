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
from activitygraphs.tensors import LocationTensors

RWPE_WALK_LENGTH = 20


@dataclass(frozen=True)
class Locations:
    """The shared location graph of a survey, held once for every user.

    Attributes:
        x: Location features, ``[num_nodes, num_location_features]``.
        edge_index: Edge list, ``[2, num_edges]``.
        edge_attr: Edge features, ``[num_edges, num_edge_features]``.
        random_walk_pe: Random-walk positional encoding of each node, ``[num_nodes, num_rwpe_features]``.
        hop_distance: Graph-hop distance between every pair of nodes, ``inf`` for unreachable pairs,
            ``[num_nodes, num_nodes]``.
        distance_m: Distance between the centroids of every pair of nodes in metres, never scaled,
            ``[num_nodes, num_nodes]``. Row ``i`` holds the distances from node ``i`` to every node.
        distance: Scaled and transformed ``distance_m`, used as a model feature, ``[num_nodes, num_nodes]``.
        node_ids: Survey ``loc_id`` of each node, one row per node.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    random_walk_pe: torch.Tensor
    hop_distance: torch.Tensor
    distance_m: torch.Tensor
    distance: torch.Tensor
    node_ids: pl.DataFrame

    @classmethod
    def from_tensors(cls, tensors: LocationTensors, transforms: FeatureTransforms | None) -> Self:
        """Return the locations of ``tensors``, with the features scaled by ``transforms``.

        Args:
            tensors: The location tensors.
            transforms: The transforms fitted on the training split. ``None`` leaves the features unscaled.
        """
        graph = pyg.data.Data(edge_index=tensors.edge_index, num_nodes=tensors.num_nodes)
        random_walk_pe = T.AddRandomWalkPE(walk_length=RWPE_WALK_LENGTH)(graph).random_walk_pe
        # T.AddLaplacianEigenvectorPE(k=8, attr_name=None), # Removed due to instability on different graphs.
        hop_distance = compute_hop_distance(tensors.edge_index, tensors.num_nodes)

        if transforms is None:
            x, edge_attr, distance = tensors.x, tensors.edge_attr, tensors.distance_m
        else:
            distance_m = tensors.distance_m.unsqueeze(-1)
            x = transforms.groups["locations"].apply(tensors.x)
            edge_attr = transforms.groups["edges"].apply(tensors.edge_attr)
            distance = transforms.groups["distances"].apply(distance_m).squeeze(-1)

        return cls(
            x=x,
            edge_index=tensors.edge_index,
            edge_attr=edge_attr,
            random_walk_pe=random_walk_pe,
            hop_distance=hop_distance,
            distance_m=tensors.distance_m,
            distance=distance,
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
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                moved[field.name] = value.to(device)

        return replace(self, **moved)


def compute_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the all-pairs contiguity-hop distance matrix ``[num_nodes, num_nodes]``, float32.

    Treats the graph as undirected and unweighted; unreachable pairs are ``inf``.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))
    hop_distance = shortest_path(adjacency, method="D", unweighted=True, directed=False)

    return torch.as_tensor(hop_distance, dtype=torch.float)
