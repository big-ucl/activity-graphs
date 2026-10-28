"""Front end for model inputs. Has one sub-encoder per input source, concatenated by ``FeatureEncoder``."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.dataset import is_home_node_mask
from activitygraphs.schema import TensorSchema


def _compute_node_index_within_graph(batch: UserBatch, device: torch.device) -> torch.Tensor:
    """Index of each node within the graph (0..N-1), aligned to the node row order of the batch."""
    return torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]


def compute_home_hops(batch: UserBatch, home_hop_distance: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Hop distance from each node to the user's home node in the graph, shape ``[num_nodes]``."""
    device = batch.x_user_node.device
    node_idx = _compute_node_index_within_graph(batch, device)
    is_home = is_home_node_mask(batch.x_user_node, is_home_idx)

    home = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
    home[batch.batch[is_home]] = node_idx[is_home]

    return home_hop_distance[home[batch.batch], node_idx]


class FeatureSource(torch.nn.Module):
    """Class that extracts one input feature source from the batch.

    Attributes:
        out_dim: Number of columns returned by ``forward``.
    """

    out_dim: int

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the columns of this source for every node of ``batch``, ``[num_nodes, out_dim]``."""
        raise NotImplementedError


class LocationFeatures(FeatureSource):
    """Extracts the location features, ``batch.x_loc``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, batch: UserBatch) -> Tensor:
        """Return ``batch.x_loc``."""
        return batch.x_loc


class UserNodeFeatures(FeatureSource):
    """Extracts the per-user node features, ``batch.x_user_node``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, batch: UserBatch) -> Tensor:
        """Return ``batch.x_user_node``."""
        return batch.x_user_node


class RandomWalkPE(FeatureSource):
    """Extracts the random-walk positional encoding, ``batch.random_walk_pe``."""

    def __init__(self, width: int):
        super().__init__()
        self.out_dim = width

    def forward(self, batch: UserBatch) -> Tensor:
        """Return ``batch.random_walk_pe``."""
        return batch.random_walk_pe


class HomeAnchoredPE(FeatureSource):
    """Extracts the home-anchored positional encoding: the RBF expansion of each node's hop distance to the user's home.

    Args:
        home_hop_distance: ``[num_nodes, num_nodes]`` contiguity-hop distance matrix of the graph.
        is_home_idx: Column index of ``is_home`` in ``batch.x_user_node``.
        bins: Number of RBF centres, spread evenly over ``[0, max_hops]``. It is also ``out_dim``.
        max_hops: Hop distance at which the encoding is capped, and the position of the last centre.
    """

    home_hop_distance: torch.Tensor

    def __init__(self, home_hop_distance: np.ndarray | torch.Tensor, is_home_idx: int, bins: int, max_hops: int):
        super().__init__()
        hop_distance = torch.as_tensor(home_hop_distance, dtype=torch.float)
        self.register_buffer("home_hop_distance", hop_distance, persistent=False)

        self.is_home_idx = is_home_idx
        self.bins = bins
        self.max_hops = max_hops
        self.out_dim = bins

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the home-anchored encoding of each node, ``[num_nodes, bins]``."""
        hops = compute_home_hops(batch, self.home_hop_distance, self.is_home_idx).clamp(max=self.max_hops)
        centers = torch.linspace(0, self.max_hops, self.bins, device=batch.x_user_node.device)

        return torch.exp(-((hops.unsqueeze(-1) - centers) ** 2) / 2.0)


class Demographics(FeatureSource):
    """Extracts the user attributes, ``batch.user_attr``, broadcast to every node of each user."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the attributes of each node's user, ``[num_nodes, out_dim]``."""
        return batch.user_attr[batch.batch]


class HomeDistance(FeatureSource):
    """Each node's distance from the user's home as fed to the model, ``batch.distances``."""

    def __init__(self, schema: TensorSchema):
        super().__init__()
        self.schema = schema
        self.out_dim = schema.width

    def forward(self, batch: UserBatch) -> Tensor:
        """Return ``batch.distances``."""
        return batch.distances


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

    def forward(self, batch: UserBatch) -> Tensor:
        """Return the model input for a given ``batch``, shape ``[num_nodes, out_dim]``."""
        source_inputs = [source(batch) for source in self.sources]
        return torch.cat(source_inputs, dim=-1)
