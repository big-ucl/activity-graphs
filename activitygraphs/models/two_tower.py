"""Two-tower scorer: a node tower run once over the location graph, a user tower, a pairwise block and a location bias.

``logit(i, n) = u_i . z_n + MLP([z_n, u_i, p(i, n)]) + b_n``, with ``z_n`` the node embedding, ``u_i`` the user
embedding, ``p(i, n)`` the pairwise features of the user and the node, and ``b_n`` the location bias.
"""

import torch
from torch import Tensor

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.features import FeatureEncoder
from activitygraphs.models.schema_encoder import SchemaEncoder
from activitygraphs.models.scorers import Scorer


class NodeTower(torch.nn.Module):
    """Embeds every node of the location graph from its features, with no information about any user.

    Args:
        network: A GNN with ``forward(x, edge_index, edge_attr, batch) -> [num_nodes, d]``, such as ``GAT``.
        num_rwpe_features: Width of the random-walk positional encoding appended to the location features, through a
            batch norm of each column. None leaves the encoding out.
    """

    def __init__(self, network: torch.nn.Module, num_rwpe_features: int | None):
        super().__init__()
        self.network = network
        self.rwpe_norm = None if num_rwpe_features is None else torch.nn.BatchNorm1d(num_rwpe_features)

    def forward(self, locations: Locations) -> Tensor:
        """Return the embedding of every node, ``[num_nodes, d]``."""
        x = locations.x
        if self.rwpe_norm is not None:
            random_walk_pe = self.rwpe_norm(locations.random_walk_pe)
            x = torch.cat([x, random_walk_pe], dim=-1)

        return self.network(x, locations.edge_index, locations.edge_attr, None)


class UserTower(torch.nn.Module):
    """Embeds each user from the user's attributes and the node embedding of the user's home.

    Args:
        network: Maps the user input, ``[num_users, in_dim]``, to the user embedding, ``[num_users, d]``, such as
            ``NodeMLP``.
        attributes: Encodes the user attributes. None leaves them out, so the home embedding is the only input.
    """

    def __init__(self, network: torch.nn.Module, attributes: SchemaEncoder | None):
        super().__init__()
        self.network = network
        self.attributes = attributes

    def forward(self, batch: UserBatch, home_embedding: Tensor) -> Tensor:
        """Return the embedding of every user of ``batch``, ``[num_users, d]``.

        Args:
            batch: The batch of users.
            home_embedding: Node embedding of each user's home, ``[num_users, d]``.
        """
        x = home_embedding
        if self.attributes is not None:
            attributes = self.attributes(batch.attributes, batch.attributes_cat)
            x = torch.cat([attributes, x], dim=-1)

        return self.network(x, None)


class LocationBias(torch.nn.Module):
    """The bias of each node, a linear function of the node's embedding.

    Args:
        embedding_dim: Width ``d`` of the node embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)

    def forward(self, node_embedding: Tensor) -> Tensor:
        """Return the bias of every node, ``[num_nodes]``, from the node embeddings, ``[num_nodes, d]``."""
        return self.linear(node_embedding).squeeze(-1)


class TwoTowerScorer(Scorer):
    """Scores each (user, node) pair by the dot product of the two embeddings, a pairwise block and a location bias.

    The node tower runs once per batch, over the single location graph.

    Args:
        node_tower: Embeds every node, ``[num_nodes, d]``.
        user_tower: Embeds every user of a batch, ``[num_users, d]``.
        pair_encoder: Builds the pairwise features of every (user, node) pair, ``[num_users, num_nodes, out_dim]``.
            None gives the pairwise block the two embeddings alone.
        pair_network: Maps the pairwise input of each pair, ``[num_pairs, 2 * d + pair_encoder.out_dim]``, to its
            logit, ``[num_pairs, 1]``, such as ``NodeMLP``.
        location_bias: The bias of every node, from the node embeddings.
    """

    def __init__(
        self,
        node_tower: NodeTower,
        user_tower: UserTower,
        pair_encoder: FeatureEncoder | None,
        pair_network: torch.nn.Module,
        location_bias: LocationBias,
    ):
        super().__init__()
        self.node_tower = node_tower
        self.user_tower = user_tower
        self.pair_encoder = pair_encoder
        self.pair_network = pair_network
        self.location_bias = location_bias

    def compile_network(self) -> None:
        """Compile the node tower's network and the pairwise network with ``torch.compile``."""
        self.node_tower.network.compile(dynamic=True)
        self.pair_network.compile(dynamic=True)

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the logit of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        num_users, num_nodes = batch.num_users, batch.num_nodes

        node_embedding = self.node_tower(locations)
        home_embedding = node_embedding[batch.home_node]
        user_embedding = self.user_tower(batch, home_embedding)

        interaction = user_embedding @ node_embedding.T

        pair_inputs = [
            node_embedding.expand(num_users, -1, -1),
            user_embedding.unsqueeze(1).expand(-1, num_nodes, -1),
        ]
        if self.pair_encoder is not None:
            pair_inputs.append(self.pair_encoder(locations, batch))

        pair_x = torch.cat(pair_inputs, dim=-1)
        flat_pair_x = pair_x.reshape(num_users * num_nodes, -1)
        pair_logits = self.pair_network(flat_pair_x, None).reshape(num_users, num_nodes)

        bias = self.location_bias(node_embedding)

        return interaction + pair_logits + bias
