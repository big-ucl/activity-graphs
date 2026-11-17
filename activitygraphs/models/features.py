"""Front end for model inputs. Has one sub-encoder per input source, concatenated by ``FeatureEncoder``."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import cast

import torch
from torch import Tensor
from torch_geometric.utils import coalesce

from activitygraphs.config import HomePEConfig
from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.schema_encoder import SchemaEncoder
from activitygraphs.schema import DatasetSchema, EdgeKind, TensorSchema
from activitygraphs.tensors import EdgeTensors


@dataclass(frozen=True)
class InputFeatures:
    """The input features used by a variant.

    Attributes:
        use_home_pe: Append the home-anchored positional encoding to each node's features.
        use_dist: Append each node's distance from home to its features.
        use_demographics: Concatenate the user attributes onto each node's features.
        use_rwpe: Append the random-walk positional encoding to each node's features.
        use_access: Append the accessibility features of each node to its features.
        use_network_hops: Instead of counting using contiguity edges in ``home_pe``, use all edge types (contiguity, rail) .
        use_tt: Append the travel time and reachability by each mode from home to each node's features.
        use_contiguity_edges: Pass messages over the contiguity edges. Ignored for models without message passing.
            Other uses (RWPE, home-anchored PE, evaluation, ...) still read the contiguity edges.
        use_rail_edges: Pass messages over the rail edges. Ignored for models without message passing.
            This flag is ignored by the home-anchored PE, which follows ``use_network_hops``.
    """

    use_home_pe: bool = True
    use_dist: bool = True
    use_demographics: bool = True
    use_rwpe: bool = True
    use_access: bool = True
    use_network_hops: bool = True
    use_tt: bool = True
    use_contiguity_edges: bool = True
    use_rail_edges: bool = True

    @property
    def departures(self) -> list[str]:
        """Return the names of the flags that differ from their default, in field order."""
        return [flag.name for flag in fields(self) if getattr(self, flag.name) != flag.default]


class FeatureSource(torch.nn.Module):
    """Class that extracts one input feature source from the locations and a batch of users.

    Attributes:
        out_dim: Number of columns returned by ``forward``.
    """

    out_dim: int

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the columns of this source for every node of every user, ``[num_users, num_nodes, out_dim]``."""
        raise NotImplementedError


class NodeFeatureSource(FeatureSource):
    """Class that extracts one input feature source that depends only on the node, not users."""

    def encode(self, locations: Locations) -> Tensor:
        """Return the columns of this source for every node, ``[num_nodes, out_dim]``."""
        raise NotImplementedError

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the columns of this source for every node of every user, ``[num_users, num_nodes, out_dim]``."""
        return self.encode(locations).expand(batch.num_users, -1, -1)


class LocationFeatures(NodeFeatureSource):
    """Extracts the location features, ``locations.x``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def encode(self, locations: Locations) -> Tensor:
        """Return ``locations.x``, ``[num_nodes, out_dim]``."""
        return locations.x


class AccessFeatures(NodeFeatureSource):
    """Extracts the accessibility features of the locations, ``locations.access``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def encode(self, locations: Locations) -> Tensor:
        """Return ``locations.access``, ``[num_nodes, out_dim]``."""
        return locations.access


class ProjectedLocationFeatures(NodeFeatureSource):
    """Projects the location features, ``locations.x``, through a city-specific linear layer.

    Args:
        schema: The column schema of the location features.
        out_dim: Width of the projection.
    """

    def __init__(self, schema: TensorSchema, out_dim: int):
        super().__init__()
        self.schema = schema
        self.projection = torch.nn.Linear(schema.width, out_dim)
        self.out_dim = out_dim

    def encode(self, locations: Locations) -> Tensor:
        """Return the projected location features, ``[num_nodes, out_dim]``."""
        return self.projection(locations.x)


class UserNodeFeatures(FeatureSource):
    """Extracts the per-user node features: ``batch.is_home``, the only column of the ``user_nodes`` group.

    Raises:
        ValueError: If ``schema`` holds any column but ``is_home``.
    """

    def __init__(self, schema: TensorSchema):
        super().__init__()
        if schema.names != ["is_home"]:
            raise ValueError(f"The per-user node features are `is_home` alone, not {schema.names}.")

        self.schema = schema
        self.out_dim = schema.width

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return ``batch.is_home`` in the dtype of ``locations.x``, ``[num_users, num_nodes, 1]``."""
        return batch.is_home.to(locations.x.dtype).unsqueeze(-1)


class RandomWalkPE(NodeFeatureSource):
    """Encodes the random-walk positional encoding, ``locations.random_walk_pe``, through a batch norm of each column."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(width)
        self.out_dim = width

    def encode(self, locations: Locations) -> Tensor:
        """Return the batch-normed encoding of each node, ``[num_nodes, out_dim]``."""
        return self.norm(locations.random_walk_pe)


class NodeEmbedding(NodeFeatureSource):
    """A learned embedding of each node's identity.

    Args:
        num_nodes: Number of nodes of the location graph.
        width: Width of the embedding.
    """

    def __init__(self, num_nodes: int, width: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, width)
        self.out_dim = width

    def encode(self, locations: Locations) -> Tensor:
        """Return the embedding of each node, ``[num_nodes, out_dim]``."""
        return self.embedding.weight


class HomeAnchoredPE(FeatureSource):
    """Extracts the home-anchored positional encoding: the RBF expansion of each node's hop distance to the user's home.

    Args:
        bins: Number of RBF centres, at least 2, spread evenly over ``[0, max_hops]``. It is also ``out_dim``. The
            width of each RBF is the spacing between two centres.
        max_hops: Hop distance at which the encoding is capped, and the position of the last centre.
        network_hops: Read the hop distance over the contiguity and rail edges, ``batch.home_network_hops``, instead of
            only the contiguity edges, ``batch.home_hops``.

    Raises:
        ValueError: ``bins`` is below 2.
    """

    def __init__(self, bins: int, max_hops: int, network_hops: bool):
        super().__init__()
        if bins < 2:
            raise ValueError(f"The home-anchored PE needs at least 2 bins, got {bins}.")

        self.bins = bins
        self.max_hops = max_hops
        self.network_hops = network_hops
        self.out_dim = bins

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the home-anchored encoding of each node for every user, ``[num_users, num_nodes, bins]``."""
        home_hops = batch.home_network_hops if self.network_hops else batch.home_hops
        hops = home_hops.clamp(max=self.max_hops)
        centers = torch.linspace(0, self.max_hops, self.bins, device=hops.device)
        width = self.max_hops / (self.bins - 1)

        return torch.exp(-(((hops.unsqueeze(-1) - centers) / width) ** 2) / 2.0)


class Demographics(FeatureSource):
    """Encodes the user attributes, ``batch.attributes`` and ``batch.attributes_cat``, for every node of each user."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.encoder = SchemaEncoder(schema)
        self.out_dim = self.encoder.out_dim

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the encoded attributes of each user at every node, ``[num_users, num_nodes, out_dim]``."""
        user_inputs = self.encoder(batch.attributes, batch.attributes_cat)
        return user_inputs.unsqueeze(1).expand(-1, batch.num_nodes, -1)


class HomeDistance(FeatureSource):
    """Each node's distance from the user's home as fed to the model, ``batch.home_distance``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return ``batch.home_distance``, ``[num_users, num_nodes, 1]``."""
        return batch.home_distance.unsqueeze(-1)


class HomeTravelTimes(FeatureSource):
    """The travel times from the user's home to each node as fed to the model, ``batch.home_travel_times``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return ``batch.home_travel_times``, ``[num_users, num_nodes, out_dim]``."""
        return batch.home_travel_times


class FeatureEncoder(torch.nn.Module):
    """Encodes the model input: encodes the columns of every source, concatenated together in order.

    Attributes:
        sources: The input sources, in column order.
        out_dim: Width of the model input, the sum of the widths of the sources.
    """

    def __init__(self, sources: Sequence[FeatureSource]):
        super().__init__()
        self.sources = torch.nn.ModuleList(sources)
        self.out_dim = sum(source.out_dim for source in sources)

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the model input of every node for every user of ``batch``, ``[num_users, num_nodes, out_dim]``."""
        source_inputs = [source(locations, batch) for source in self.sources]
        return torch.cat(source_inputs, dim=-1)


class NodeFeatureEncoder(torch.nn.Module):
    """Encodes the node-level input once over the graph: the columns of every node source, concatenated in order.

    Attributes:
        sources: The node sources, in column order.
        out_dim: Width of the input, the sum of the widths of the sources.
    """

    def __init__(self, sources: Sequence[NodeFeatureSource]):
        super().__init__()
        self.sources = torch.nn.ModuleList(sources)
        self.out_dim = sum(source.out_dim for source in sources)

    def forward(self, locations: Locations) -> Tensor:
        """Return the node-level input of every node, ``[num_nodes, out_dim]``."""
        node_sources = cast(Sequence[NodeFeatureSource], self.sources)
        source_inputs = [source.encode(locations) for source in node_sources]
        return torch.cat(source_inputs, dim=-1)


class EdgeSource(torch.nn.Module):
    """Extracts the edges of one kind, ``locations.edges[kind]``.

    Attributes:
        kind: The kind of the edges.
        out_dim: Number of features of each edge.
    """

    def __init__(self, kind: EdgeKind, schema: TensorSchema):
        super().__init__()
        self.kind = kind
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, locations: Locations) -> EdgeTensors:
        """Return ``locations.edges[kind]``, with ``out_dim`` features per edge."""
        return locations.edges[self.kind]


class EdgeEncoder(torch.nn.Module):
    """Encodes the edges for message passing: the edges of every source, merged into one edge per pair of nodes.

    A single source keeps its own edges and features. Several sources are merged by ``_merge_edges``, with a flag per
    source then the features of each source. No source leaves every node with no neighbour.

    Attributes:
        sources: The edge sources, in column order.
        out_dim: Number of features of each encoded edge.
    """

    def __init__(self, sources: Sequence[EdgeSource]):
        super().__init__()
        self.sources = torch.nn.ModuleList(sources)
        num_flags = 0 if len(sources) == 1 else len(sources)
        self.out_dim = num_flags + sum(source.out_dim for source in sources)

    def forward(self, locations: Locations) -> EdgeTensors:
        """Return the encoded edges of the location graph, with ``out_dim`` features per edge."""
        source_edges = [source(locations) for source in self.sources]
        if len(source_edges) == 1:
            return source_edges[0]

        return _merge_edges(source_edges, locations.num_nodes, locations.x.device)


def build_feature_encoder(
    features: InputFeatures, locations: Locations, schema: DatasetSchema, home_pe_cfg: HomePEConfig
) -> FeatureEncoder:
    """Instantiate the input ``FeatureEncoder`` for the associated ``features`` and the survey ``locations``.

    The column order goes:
      1. node features (location and accessibility features and random-walk PE), as in ``build_node_feature_encoder``,
      2. per-user node features,
      3. home-anchored PE, over the network hops when ``features.use_network_hops`` is set,
      4. demographics,
      5. distance from home, and
      6. travel times from home.

    Args:
        features: The input features to encode.
        locations: The location graph the model is trained on.
        schema: The column schema of each tensor group.
        home_pe_cfg: Settings of the home-anchored positional encoding, read when ``features.use_home_pe`` is set.
    """
    sources: list[FeatureSource] = [
        *_node_sources(features, locations, schema),
        UserNodeFeatures(schema.user_nodes),
    ]

    if features.use_home_pe:
        sources.append(HomeAnchoredPE(home_pe_cfg.bins, home_pe_cfg.max_hops, features.use_network_hops))
    if features.use_demographics:
        sources.append(Demographics(schema.user_attributes))
    if features.use_dist:
        sources.append(HomeDistance(schema.distances))
    if features.use_tt:
        sources.append(HomeTravelTimes(schema.travel_times))

    return FeatureEncoder(sources)


def build_node_feature_encoder(
    features: InputFeatures, locations: Locations, schema: DatasetSchema, node_embedding_dim: int | None
) -> NodeFeatureEncoder:
    """Instantiate the ``NodeFeatureEncoder`` of the node-level input selected by ``features``.

    The column order goes: location features, accessibility features, random-walk PE, then node embedding.

    Args:
        features: The input features to encode.
        locations: The location graph the model is trained on.
        schema: The column schema of each tensor group.
        node_embedding_dim: Width of a learned embedding of each node's identity. None leaves it out.
    """
    sources = _node_sources(features, locations, schema)
    if node_embedding_dim is not None:
        sources.append(NodeEmbedding(locations.num_nodes, node_embedding_dim))

    return NodeFeatureEncoder(sources)


def build_edge_encoder(features: InputFeatures, schema: DatasetSchema) -> EdgeEncoder:
    """Instantiate the ``EdgeEncoder`` of the edges selected by ``features``.

    The column order goes: contiguity edges, then rail edges.

    Args:
        features: The input features to encode.
        schema: The column schema of each tensor group.
    """
    sources = []
    if features.use_contiguity_edges:
        sources.append(EdgeSource(EdgeKind.CONTIGUITY, schema.edges[EdgeKind.CONTIGUITY]))
    if features.use_rail_edges:
        sources.append(EdgeSource(EdgeKind.RAIL, schema.edges[EdgeKind.RAIL]))

    return EdgeEncoder(sources)


def _node_sources(features: InputFeatures, locations: Locations, schema: DatasetSchema) -> list[NodeFeatureSource]:
    """Return the node-level sources selected by ``features``, in column order.

    The column order goes: location features, accessibility features, then random-walk PE.

    Args:
        features: The input features to encode.
        locations: The location graph the model is trained on.
        schema: The column schema of each tensor group.
    """
    sources: list[NodeFeatureSource] = [LocationFeatures(schema.locations)]
    if features.use_access:
        sources.append(AccessFeatures(schema.access))
    if features.use_rwpe:
        sources.append(RandomWalkPE(locations.num_rwpe_features))

    return sources


def _merge_edges(edges: Sequence[EdgeTensors], num_nodes: int, device: torch.device) -> EdgeTensors:
    """Return the edges of ``edges`` merged into one edge per ordered pair of nodes, with a flag per edge set.

    Args:
        edges: The edge sets to merge, each pair of nodes at most once per set.
        num_nodes: Number of nodes of the graph.
        device: Device of the edges.

    Returns:
        The merged edges, sorted by source node and then by target node. The features are a flag per edge set, followed
        by the features of each set. The flag of a set is 1 where the pair has an edge in that set, and 0 elsewhere. The
        columns of a set hold the features of that edge, and 0 where the pair has no edge in that set.
    """
    widths = [edge_set.attr.shape[1] for edge_set in edges]
    num_columns = len(edges) + sum(widths)
    index_blocks = [torch.zeros(2, 0, dtype=torch.long, device=device)]
    attr_blocks = [torch.zeros(0, num_columns, device=device)]

    first_column = len(edges)
    for set_position, (edge_set, width) in enumerate(zip(edges, widths, strict=True)):
        attr_block = torch.zeros(edge_set.num_edges, num_columns, device=device)
        attr_block[:, set_position] = 1.0
        attr_block[:, first_column : first_column + width] = edge_set.attr
        index_blocks.append(edge_set.index)
        attr_blocks.append(attr_block)
        first_column += width

    stacked_index = torch.cat(index_blocks, dim=1)
    stacked_attr = torch.cat(attr_blocks)
    index, attr = coalesce(stacked_index, stacked_attr, num_nodes=num_nodes, reduce="sum")

    return EdgeTensors(index, attr)
