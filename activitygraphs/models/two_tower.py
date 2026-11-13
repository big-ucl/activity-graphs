"""Two-tower scorer: a node tower run once over the location graph, a user tower, a pairwise block and a location bias.

``logit(i, n) = u_i . z_n + MLP([z_n, u_i, p(i, n)]) + b_n``, with ``z_n`` the node embedding, ``u_i`` the user
embedding, ``p(i, n)`` the pairwise features of the user and the node, and ``b_n`` the location bias.
"""

import torch
from torch import Tensor

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.features import FeatureEncoder, NodeFeatureEncoder
from activitygraphs.models.schema_encoder import SchemaEncoder
from activitygraphs.models.scorers import Scorer


class NodeTower(torch.nn.Module):
    """Embeds every node of the location graph from its features, with no information about any user.

    Args:
        node_encoder: Encodes the node-level input of every node.
        network: A GNN with ``forward(x, edge_index, edge_attr, batch) -> [num_nodes, d]``, such as ``GAT``.
    """

    def __init__(self, node_encoder: NodeFeatureEncoder, network: torch.nn.Module):
        super().__init__()
        self.node_encoder = node_encoder
        self.network = network

    def forward(self, locations: Locations) -> Tensor:
        """Return the embedding of every node, ``[num_nodes, d]``."""
        x = self.node_encoder(locations)
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

    def forward(self, locations: Locations, node_embedding: Tensor) -> Tensor:
        """Return the bias of every node, ``[num_nodes]``, from the node embeddings, ``[num_nodes, d]``."""
        return self.linear(node_embedding).squeeze(-1)


class MLPBias(torch.nn.Module):
    """The bias of each node, an MLP of either the node's embedding or the node's projected location features.

    Args:
        network: Maps the input of each node, ``[num_nodes, in_dim]``, to the node's bias, ``[num_nodes, 1]``, such
            as ``NodeMLP``.
        projection: City-specific projection of the location features, ``locations.x``, to the input of
            ``network``. None gives ``network`` the node embeddings.
    """

    def __init__(self, network: torch.nn.Module, projection: torch.nn.Linear | None):
        super().__init__()
        self.network = network
        self.projection = projection

    def forward(self, locations: Locations, node_embedding: Tensor) -> Tensor:
        """Return the bias of every node, ``[num_nodes]``.

        Args:
            locations: The location graph, whose features are read when ``projection`` is set.
            node_embedding: The node embeddings, ``[num_nodes, d]``, read when ``projection`` is None.
        """
        x = node_embedding if self.projection is None else self.projection(locations.x)

        return self.network(x, None).squeeze(-1)


class NodeBias(torch.nn.Module):
    """The bias of each node, a free parameter per node, starting at zero.

    It cannot score a node of another graph, so it is an in-city diagnostic.

    Args:
        num_nodes: Number of nodes of the graph.
    """

    def __init__(self, num_nodes: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_nodes))

    def forward(self, locations: Locations, node_embedding: Tensor) -> Tensor:
        """Return the bias of every node, ``[num_nodes]``. The locations and the node embeddings are not read."""
        return self.bias


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
        bias: The bias of every node, such as ``LocationBias``, ``MLPBias`` or ``NodeBias``.
    """

    def __init__(
        self,
        node_tower: NodeTower,
        user_tower: UserTower,
        pair_encoder: FeatureEncoder | None,
        pair_network: torch.nn.Module,
        bias: LocationBias | MLPBias | NodeBias,
    ):
        super().__init__()
        self.node_tower = node_tower
        self.user_tower = user_tower
        self.pair_encoder = pair_encoder
        self.pair_network = pair_network
        self.bias = bias

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

        bias = self.bias(locations, node_embedding)

        return interaction + pair_logits + bias
