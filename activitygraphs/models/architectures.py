"""The architecture registry: maps the ``architecture`` key of a model config to the builder of its scorer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from activitygraphs.config import GATConfig, GPSConfig, HomePEConfig, HomeZoneMFConfig, MLPConfig, ModelConfig
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.baselines import HomeZoneMFBaseline
from activitygraphs.models.features import FeatureEncoder, InputFeatures, build_feature_encoder
from activitygraphs.models.networks import GAT, GraphTransformer, NodeMLP
from activitygraphs.models.scorers import EncodedScorer, GraphScorer, NodeScorer, Scorer
from activitygraphs.schema import DatasetSchema

type ScorerBuilder = Callable[[ModelConfig, InputFeatures, Locations, DatasetSchema, HomePEConfig], Scorer]


@dataclass(frozen=True)
class NetworkSizes:
    """Input and output sizes of a network, set by the dataset and the variant's input features.

    Attributes:
        in_channels: Width of each node's feature vector.
        out_channels: Width of each node's output.
        edge_dim: Width of each edge's feature vector.
    """

    in_channels: int
    out_channels: int
    edge_dim: int


def build_mlp(config: MLPConfig, sizes: NetworkSizes) -> NodeMLP:
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


def build_gat(config: GATConfig, sizes: NetworkSizes) -> GAT:
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


def build_gps(config: GPSConfig, sizes: NetworkSizes) -> GraphTransformer:
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


def encoded_scorer(
    build_network_fn: Callable[..., torch.nn.Module],
    scorer_cls: type[EncodedScorer],
) -> ScorerBuilder:
    """Return the builder for a scorer that encodes network inputs with ``FeatureEncoder``.

    The builder sizes the network based on the encoder's ``out_dim`` and the widths of the schema, and wraps the encoder
    and builds the scorer with ``scorer_cls(encoder, network)``. Every input feature is supported.

    Args:
        build_network_fn: Builds the network from its config and its ``NetworkSizes``.
        scorer_cls: The scorer class, constructed from the ``FeatureEncoder`` and the network.
    """

    def build(
        config: ModelConfig,
        features: InputFeatures,
        locations: Locations,
        schema: DatasetSchema,
        home_pe_cfg: HomePEConfig,
    ) -> Scorer:
        encoder = build_feature_encoder(features, locations, schema, home_pe_cfg)

        sizes = NetworkSizes(
            in_channels=encoder.out_dim,
            out_channels=schema.targets.width,
            edge_dim=schema.edges.width,
        )
        network = build_network_fn(config, sizes)

        return scorer_cls(encoder, network)

    return build


def build_home_zone_mf(
    config: ModelConfig,
    features: InputFeatures,
    locations: Locations,
    schema: DatasetSchema,
    home_pe_cfg: HomePEConfig,
) -> HomeZoneMFBaseline:
    """Instantiate a ``HomeZoneMFBaseline`` over the nodes of ``locations``.

    Raises:
        ValueError: If ``features`` differs from the defaults. The scorer reads no input feature.
    """
    if features != InputFeatures():
        raise ValueError(f"HomeZoneMF reads no input feature, so its flags must keep their defaults, got {features}")

    home_zone_mf = cast(HomeZoneMFConfig, config)

    return HomeZoneMFBaseline(locations.num_nodes, home_zone_mf.n_factors)


BUILDERS: dict[str, ScorerBuilder] = {
    "mlp": encoded_scorer(build_mlp, NodeScorer),
    "gat": encoded_scorer(build_gat, GraphScorer),
    "gps": encoded_scorer(build_gps, GraphScorer),
    "home_zone_mf": build_home_zone_mf,
}


def lookup_architecture(config: ModelConfig) -> ScorerBuilder:
    """Return the scorer builder registered in ``BUILDERS`` under ``config.architecture``.

    Raises:
        KeyError: If no architecture is registered under ``config.architecture``.
    """
    if config.architecture not in BUILDERS:
        raise KeyError(f"No architecture registered under {config.architecture!r}, expected one of {sorted(BUILDERS)}")

    return BUILDERS[config.architecture]


def build_scorer(
    config: ModelConfig,
    features: InputFeatures,
    locations: Locations,
    schema: DatasetSchema,
    home_pe_cfg: HomePEConfig,
) -> Scorer:
    """Instantiate the scorer of the architecture ``config``, with the input ``features``.

    Args:
        config: The architecture, its optimiser settings and its hyperparameters, from ``cfg.models``.
        features: The input features the scorer is trained with.
        locations: The location graph the scorer is trained on.
        schema: The column schema of each tensor group.
        home_pe_cfg: Settings of the home-anchored positional encoding, read when ``features.use_home_pe`` is set.
    """
    build = lookup_architecture(config)

    return build(config, features, locations, schema, home_pe_cfg)
