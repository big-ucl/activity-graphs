"""PyTorch GNN architectures: GATSkip (primary), GraphTransformer, NodeMLP, and helpers."""

import functools
import itertools

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn.conv import GCNConv, GPSConv, GATConv


def build_module_list(
    module_f: torch.nn.Module,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int | None = None,
    **module_kwargs,
) -> torch.nn.ModuleList:
    """Construct a ``ModuleList`` of ``num_layers`` stacked ``module_f`` instances.

    Args:
        module_f: Callable ``(in, out, **kwargs) → nn.Module`` used to create each layer.
        num_layers: Total number of layers; must be ≥ 1.
        in_channels: Input size of the first layer.
        out_channels: Output size of the last layer.
        hidden_channels: Hidden size for intermediate layers; required when ``num_layers > 1``.
        **module_kwargs: Extra keyword arguments forwarded to ``module_f``.

    Returns:
        ``ModuleList`` of length ``num_layers``.
    """
    if num_layers > 1 and hidden_channels is None:
        raise ValueError(f"Hidden channels not provided for {num_layers=} and {hidden_channels=}")

    if num_layers < 1:
        raise ValueError(f"Invalid number of layers: {num_layers=}")

    layers = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
    modules = torch.nn.ModuleList()

    for num_in, num_out in itertools.pairwise(layers):
        modules.append(module_f(num_in, num_out, **module_kwargs))

    return modules


class GCN(torch.nn.Module):
    """Multi-layer graph convolutional network with optional residual connections and leaky-ReLU activations."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        residuals: bool = False,
        conv: pyg.nn.conv.MessagePassing = None,
    ):
        super().__init__()

        conv = GCNConv if conv is None else conv

        if residuals and in_channels != hidden_channels:
            raise ValueError("Number of in channels must match number of hidden channels")

        self.residuals = residuals
        self.dropout = dropout
        self.convs = build_module_list(conv, num_layers, in_channels, out_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x_res = x
            x = F.dropout(x, p=self.dropout, training=self.training)

            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            if self.residuals:
                x = x + x_res

            x = F.leaky_relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)

        return x


class NodeMLP(torch.nn.Module):
    """Node-wise MLP with dropout and ReLU activations."""

    def __init__(
        self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2
    ):
        super().__init__()

        self.dropout = dropout
        self.lins = build_module_list(torch.nn.Linear, num_layers, in_channels, out_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        for lin in self.lins[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x).relu()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


class FullyConnectedMLP(torch.nn.Module):
    """Per-user global MLP: flattens the whole graph's node features and predicts every node at once.

    Tries to serve as an "upper bound" on what is doable with a GNN. The idea is that a GNN approaches
    this baseline without needing to have a fixed-size node graph. It relies on every user graph having
    exactly ``num_nodes`` nodes (true for the shared network graph). The constant ``edge_index`` /
    ``edge_attr`` carry no per-user signal and are ignored.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_features = in_features
        self.mlp = NodeMLP(
            num_layers,
            in_channels=num_nodes * in_features,
            hidden_channels=hidden_channels,
            out_channels=num_nodes,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        total_nodes, num_features = x.shape

        if total_nodes % self.num_nodes != 0:
            raise ValueError(f"Expected a multiple of {self.num_nodes} nodes per batch, got {total_nodes}.")
        if num_features != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {num_features}.")

        num_graphs = total_nodes // self.num_nodes

        flat = x.reshape(num_graphs, self.num_nodes * num_features)
        logits = self.mlp(flat, edge_index)  # [num_graphs, num_nodes]

        return logits.reshape(total_nodes, 1)


class GCNPlus(torch.nn.Module):
    """GCN followed by a node-wise MLP post-processing head."""

    def __init__(
        self, num_gcn: int, num_lin: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2
    ):
        super().__init__()

        self.gcn = GCN(num_gcn, in_channels, hidden_channels, hidden_channels, dropout)
        self.lin = NodeMLP(num_lin, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        x = self.gcn(x, edge_index, edge_attr).relu()
        x = self.lin(x, edge_index)

        return x


class GCNRes(torch.nn.Module):
    """GCN with separate MLP pre-/post-processing heads and residual connections in the GCN."""

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        self.convs = GCN(num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=True)
        self.post_lin = NodeMLP(num_post_layers, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        x = self.pre_lin(x, edge_index)
        x = self.convs(x, edge_index)
        x = self.post_lin(x, edge_index)

        return x


class GCNSkip(torch.nn.Module):
    """GCN with a skip connection that concatenates raw input features to the post-GCN representation."""

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        residuals: bool = False,
        conv: pyg.nn.conv.MessagePassing | None = None,
    ):
        super().__init__()

        self.dropout = dropout
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        self.convs = GCN(
            num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=residuals, conv=conv
        )
        self.post_lin = NodeMLP(num_post_layers, hidden_channels + in_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        x_skip = x

        x = self.pre_lin(x, edge_index)
        x = self.convs(x, edge_index, edge_attr)

        x = torch.cat([x, x_skip], dim=1)
        x = self.post_lin(x, edge_index)

        return x


class GATSkip(GCNSkip):
    """Primary model: ``GCNSkip`` with GAT convolutions and edge features. Preferred over ``GCNSkip``."""

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.2,
        residuals: bool = False,
    ):
        super().__init__(
            num_pre_layers,
            num_gcn_layers,
            num_post_layers,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            residuals,
            conv=functools.partial(GATConv, edge_dim=edge_dim),
        )


class GPSLayer(torch.nn.Module):
    """Single GPS layer: GAT local message-passing + Performer global attention."""

    def __init__(self, hidden_channels, edge_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.conv = GPSConv(
            channels=hidden_channels,
            conv=GATConv(hidden_channels, hidden_channels, heads=1, edge_dim=edge_dim, add_self_loops=False),
            heads=num_heads,
            dropout=dropout,
            attn_type="performer",
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.conv(x, edge_index, batch, edge_attr=edge_attr)


class GraphTransformer(torch.nn.Module):
    """Graph Transformer (GPS architecture) with linear input/edge projections and an MLP output head."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.edge_proj = torch.nn.Linear(edge_dim, hidden_channels)

        self.layers = torch.nn.ModuleList([
            GPSLayer(hidden_channels, hidden_channels, num_heads, dropout) for _ in range(num_layers)
        ])

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        return self.output(x)
