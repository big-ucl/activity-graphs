"""The architecture registry: maps the ``architecture`` key of a model config to the builder of its model."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from activitygraphs.config import GATConfig, GPSConfig, MLPConfig, ModelConfig
from activitygraphs.ml.models import GAT, GraphTransformer, NodeMLP


@dataclass(frozen=True)
class ModelSizes:
    """Input and output sizes of a model, set by the dataset and the variant's input features.

    Attributes:
        in_channels: Width of each node's feature vector.
        out_channels: Width of each node's output.
        edge_dim: Width of each edge's feature vector.
    """

    in_channels: int
    out_channels: int
    edge_dim: int


def build_mlp(config: MLPConfig, sizes: ModelSizes) -> NodeMLP:
    """Instantiate a ``NodeMLP`` from ``config``, sized by ``sizes``."""
    return NodeMLP(
        num_layers=config.num_layers,
        in_channels=sizes.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=sizes.out_channels,
        dropout=config.dropout,
    )


NORMS: dict[str, Callable[[int], torch.nn.Module]] = {
    "layer_norm": torch.nn.LayerNorm,
    "batch_norm": torch.nn.BatchNorm1d,
}


def build_gat(config: GATConfig, sizes: ModelSizes) -> GAT:
    """Instantiate a ``GAT`` from ``config``, sized by ``sizes``.

    Raises:
        KeyError: If ``config.norm`` is neither None nor a key of ``NORMS``.
    """
    if config.norm is not None and config.norm not in NORMS:
        raise KeyError(f"No normalisation registered under {config.norm!r}, expected one of {sorted(NORMS)}")

    return GAT(
        num_pre_layers=config.num_pre_layers,
        num_gcn_layers=config.num_gcn_layers,
        num_post_layers=config.num_post_layers,
        in_channels=sizes.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=sizes.out_channels,
        edge_dim=sizes.edge_dim,
        dropout=config.dropout,
        residuals=config.residuals,
        norm=None if config.norm is None else NORMS[config.norm],
    )


def build_gps(config: GPSConfig, sizes: ModelSizes) -> GraphTransformer:
    """Instantiate a ``GraphTransformer`` (GPS) from ``config``, sized by ``sizes``."""
    return GraphTransformer(
        in_channels=sizes.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=sizes.out_channels,
        edge_dim=sizes.edge_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )


BUILDERS: dict[str, Callable[..., torch.nn.Module]] = {
    "mlp": build_mlp,
    "gat": build_gat,
    "gps": build_gps,
}


def architecture_builder(config: ModelConfig) -> Callable[..., torch.nn.Module]:
    """Return the builder corresponding to ``config.architecture`` from ``BUILDERS``.

    Raises:
        KeyError: If no builder is registered under ``config.architecture``.
    """
    if config.architecture not in BUILDERS:
        raise KeyError(f"No architecture registered under {config.architecture!r}, expected one of {sorted(BUILDERS)}")

    return BUILDERS[config.architecture]
