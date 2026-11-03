"""Scorers: the models that score every node for every user of a batch, ``scorer(locations, batch) -> [B, N]``."""

from typing import Self

import torch
from torch import Tensor

from activitygraphs.datasets.batch import GraphBatch, UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.models.features import FeatureEncoder


class Scorer(torch.nn.Module):
    """A model that scores every node of the location graph for every user of a batch."""

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the logit of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        raise NotImplementedError

    def compile_network(self) -> None:
        """Compile the trained network of the scorer with ``torch.compile``. A scorer without one compiles nothing."""


class NodeScorer(Scorer):
    """Scores each (user, node) pair from its own model input alone, with a node-wise network such as ``NodeMLP``.

    Args:
        encoder: Builds the model input of every (user, node) pair.
        network: Maps the model input of each pair, ``[num_pairs, encoder.out_dim]``, to its logit, ``[num_pairs, 1]``.
    """

    def __init__(self, encoder: FeatureEncoder, network: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.network = network

    def compile_network(self) -> None:
        """Compile ``network`` with ``torch.compile``, for inputs of any batch size."""
        self.network.compile(dynamic=True)

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the logit of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        x = self.encoder(locations, batch)
        flat_x = x.reshape(-1, x.shape[-1])
        logits = self.network(flat_x, None, None, None)

        return logits.reshape(batch.num_users, batch.num_nodes)


class GraphScorer(Scorer):
    """Scores each (user, node) pair by message passing over one copy of the location graph per user.

    Args:
        encoder: Builds the model input of every (user, node) pair.
        network: A GNN with ``forward(x, edge_index, edge_attr, batch) -> [num_pairs, 1]`` logits, such as ``GAT``.
    """

    def __init__(self, encoder: FeatureEncoder, network: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.network = network

    def compile_network(self) -> None:
        """Compile ``network`` with ``torch.compile``, for inputs of any batch size."""
        self.network.compile(dynamic=True)

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the logit of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        x = self.encoder(locations, batch)
        graphs = GraphBatch.expand(x, locations)
        logits = self.network(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch)

        return logits.reshape(batch.num_users, batch.num_nodes)


class FittedScorer(Scorer):
    """A scorer fitted once on the training users, outside the training loop, and then only evaluated.

    It is scored from the batch itself, rather than from the input built by a ``FeatureEncoder``.
    """

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Fit the scorer and return it. A scorer with nothing to fit returns itself unchanged.

        Args:
            train: The training users.
            val: The validation users.
            locations: The location graph.
        """
        return self
