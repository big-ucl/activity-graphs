"""The architecture registry: maps the ``architecture`` key of a model config to the builder of its scorer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from activitygraphs.config import (
    GATConfig,
    GPSConfig,
    HomePEConfig,
    HomeZoneMFConfig,
    MLPConfig,
    ModelConfig,
    TwoTowerConfig,
)
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.baselines import HomeZoneMFBaseline
from activitygraphs.models.features import (
    FeatureEncoder,
    FeatureSource,
    HomeAnchoredPE,
    HomeDistance,
    InputFeatures,
    ProjectedLocationFeatures,
    build_feature_encoder,
    build_node_feature_encoder,
)
from activitygraphs.models.networks import GAT, GraphTransformer, NodeMLP
from activitygraphs.models.schema_encoder import SchemaEncoder
from activitygraphs.models.scorers import EncodedScorer, GraphScorer, NodeScorer, Scorer
from activitygraphs.models.two_tower import LocationBias, MLPBias, NodeBias, NodeTower, TwoTowerScorer, UserTower
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


TWO_TOWER_BIASES = ("location", "embedding_mlp", "feature_mlp", "node")

NORMS: dict[str, Callable[[int], torch.nn.Module]] = {
    "layer_norm": torch.nn.LayerNorm,
    "batch_norm": torch.nn.BatchNorm1d,
}


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
    build = _lookup_architecture(config)

    return build(config, features, locations, schema, home_pe_cfg)


def _build_mlp(config: MLPConfig, sizes: NetworkSizes) -> NodeMLP:
    """Instantiate a ``NodeMLP`` from ``config``, sized by ``sizes``."""
    return NodeMLP(
        num_layers=config.num_layers,
        in_channels=sizes.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=sizes.out_channels,
        dropout=config.dropout,
    )


def _build_gat(config: GATConfig, sizes: NetworkSizes) -> GAT:
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


def _build_gps(config: GPSConfig, sizes: NetworkSizes) -> GraphTransformer:
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


def _encoded_scorer(
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


def _build_home_zone_mf(
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


def _build_two_tower(
    config: ModelConfig,
    features: InputFeatures,
    locations: Locations,
    schema: DatasetSchema,
    home_pe_cfg: HomePEConfig,
) -> TwoTowerScorer:
    """Instantiate a ``TwoTowerScorer`` for the input ``features``.

    Features inputs are given to the model components as follows:
    - Node tower: accessibility features (``use_access``) and random-walk PE (``use_rwpe``).
    - User tower: user attributes (``use_demographics``)
    - Pairwise block: home-anchored PE (``use_home_pe``) and distance from home (``use_dist``).

    Raises:
        ValueError: If ``config.bias`` is not one of ``TWO_TOWER_BIASES``.
    """
    two_tower = cast(TwoTowerConfig, config)
    embedding_dim = two_tower.embedding_dim
    hidden_channels = two_tower.hidden_channels
    dropout = two_tower.dropout
    num_nodes = locations.num_nodes

    if two_tower.bias not in TWO_TOWER_BIASES:
        raise ValueError(f"The two-tower bias is one of {TWO_TOWER_BIASES}, got {two_tower.bias!r}")

    node_embedding_dim = embedding_dim if two_tower.node_embedding else None
    node_encoder = build_node_feature_encoder(features, locations, schema, node_embedding_dim)

    node_sizes = NetworkSizes(
        in_channels=node_encoder.out_dim,
        out_channels=embedding_dim,
        edge_dim=schema.edges.width,
    )
    node_network = _build_gat(two_tower, node_sizes)
    node_tower = NodeTower(node_encoder, node_network)

    attributes = SchemaEncoder(schema.user_attributes) if features.use_demographics else None
    user_in_channels = embedding_dim + (0 if attributes is None else attributes.out_dim)
    user_network = NodeMLP(two_tower.num_user_layers, user_in_channels, hidden_channels, embedding_dim, dropout)
    user_tower = UserTower(user_network, attributes)

    pair_sources: list[FeatureSource] = []
    if two_tower.pair_location_features:
        pair_sources.append(ProjectedLocationFeatures(schema.locations, embedding_dim))
    if features.use_home_pe:
        pair_sources.append(HomeAnchoredPE(home_pe_cfg.bins, home_pe_cfg.max_hops))
    if features.use_dist:
        pair_sources.append(HomeDistance(schema.distances))

    pair_encoder = FeatureEncoder(pair_sources) if pair_sources else None
    pair_in_channels = 2 * embedding_dim + (0 if pair_encoder is None else pair_encoder.out_dim)
    pair_network = NodeMLP(two_tower.num_pair_layers, pair_in_channels, hidden_channels, 1, dropout)

    bias = _build_two_tower_bias(two_tower, schema, num_nodes)

    return TwoTowerScorer(node_tower, user_tower, pair_encoder, pair_network, bias)


def _build_two_tower_bias(
    config: TwoTowerConfig, schema: DatasetSchema, num_nodes: int
) -> LocationBias | MLPBias | NodeBias:
    """Instantiate the node bias of the two-tower model selected by ``config.bias``.

    Args:
        config: The two-tower config, whose ``bias`` names one of ``TWO_TOWER_BIASES``.
        schema: The column schema of each tensor group, read for the width of the location features.
        num_nodes: Number of nodes of the location graph.
    """
    embedding_dim = config.embedding_dim

    if config.bias == "location":
        return LocationBias(embedding_dim)
    if config.bias == "node":
        return NodeBias(num_nodes)

    network = NodeMLP(config.num_bias_layers, embedding_dim, config.hidden_channels, 1, config.dropout)
    projection = torch.nn.Linear(schema.locations.width, embedding_dim) if config.bias == "feature_mlp" else None

    return MLPBias(network, projection)


def _lookup_architecture(config: ModelConfig) -> ScorerBuilder:
    """Return the scorer builder registered in ``BUILDERS`` under ``config.architecture``.

    Raises:
        KeyError: If no architecture is registered under ``config.architecture``.
    """
    if config.architecture not in BUILDERS:
        raise KeyError(f"No architecture registered under {config.architecture!r}, expected one of {sorted(BUILDERS)}")

    return BUILDERS[config.architecture]


BUILDERS: dict[str, ScorerBuilder] = {
    "mlp": _encoded_scorer(_build_mlp, NodeScorer),
    "gat": _encoded_scorer(_build_gat, GraphScorer),
    "gps": _encoded_scorer(_build_gps, GraphScorer),
    "home_zone_mf": _build_home_zone_mf,
    "two_tower": _build_two_tower,
}
