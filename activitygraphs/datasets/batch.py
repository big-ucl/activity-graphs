"""Typed view of a collated batch of per-user graphs."""

from dataclasses import dataclass, fields

import torch_geometric as pyg
from torch import Tensor


@dataclass(frozen=True)
class UserBatch:
    """The tensors of a collated ``VisitDataset`` batch, typed.

    Attributes:
        x: Node features, ``[num_nodes, num_features]``.
        edge_index: Edge list, ``[2, num_edges]``.
        edge_attr: Edge features, ``[num_edges, num_edge_features]``.
        y: Visit labels, ``[num_nodes, 1]``.
        batch: Graph index of each node, ``[num_nodes]``.
        ptr: Offset of each graph's first node, ``[num_graphs + 1]``.
        user_attr: Per-user attributes, ``[num_graphs, num_user_attributes]``.
        user_id: Dataset index of each user, ``[num_graphs]``.
        distances: Distance from the user's home to each node, ``[num_nodes, 1]``.
    """

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    batch: Tensor
    ptr: Tensor
    user_attr: Tensor
    user_id: Tensor
    distances: Tensor

    @classmethod
    def from_pyg(cls, batch: pyg.data.Batch) -> "UserBatch":
        """Return the typed view of a PyG batch collated from ``VisitDataset`` items."""
        return cls(**{field.name: getattr(batch, field.name) for field in fields(cls)})

    @property
    def num_nodes(self) -> int:
        """Number of nodes over all graphs in the batch."""
        return self.x.shape[0]

    @property
    def num_graphs(self) -> int:
        """Number of graphs (users) in the batch."""
        return self.user_id.shape[0]
