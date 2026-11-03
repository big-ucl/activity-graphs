"""Common utilies for baseline models."""

import torch
from torch_geometric.utils import to_dense_batch

from activitygraphs.datasets.batch import GraphBatch, is_home_node_mask
from activitygraphs.datasets.users import Users
from activitygraphs.ml.popularity import POPULARITY_EPS


class Baseline(torch.nn.Module):
    """A model scored from the batch itself, rather than from the model input built by the ``FeatureEncoder``."""

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        """Return the logit of each node of ``batch``, shape ``(num_nodes, 1)``."""
        raise NotImplementedError


def non_home_mask(users: Users) -> torch.Tensor:
    """Return the ``[n_users, n_nodes]`` mask of the (user, node) pairs where the node is not the user's home."""
    mask = torch.ones_like(users.visited, dtype=torch.bool)
    mask[torch.arange(len(mask)), users.home_node] = False
    return mask


def graph_node_indices(batch: GraphBatch) -> torch.Tensor:
    """Return, for each node, its within-graph index, ``[num_nodes]``."""
    return torch.arange(batch.num_nodes, device=batch.batch.device) - batch.ptr[batch.batch]


def graph_home_indices(batch: GraphBatch, is_home_idx_user_node: int) -> torch.Tensor:
    """Return, for each node, the within-graph index of the corresponding home node, ``[num_nodes]``.

    Args:
        batch: The batch.
        is_home_idx_user_node: Column index of ``is_home`` in ``batch.x_user_node``.

    Raises:
        ValueError: If a graph has no home node.
    """
    home_mask = is_home_node_mask(batch.x_user_node, is_home_idx_user_node)
    is_home, _ = to_dense_batch(home_mask, batch.batch, fill_value=False)
    if not is_home.any(dim=1).all():
        raise ValueError("No home node found.")

    return is_home.int().argmax(dim=1)[batch.batch]


SMOOTHING_GRID = torch.logspace(-2, 3, 30)


def smoothed_rate(
    counts: torch.Tensor, users: torch.Tensor, prior: torch.Tensor, strength: float | torch.Tensor
) -> torch.Tensor:
    """Compute ``(counts + strength * prior) / (users + strength)``, clamped to ``[1e-6, 1 - 1e-6]``.

    Args:
        counts: Per-home visit counts, ``[..., n_nodes]``.
        users: Residents of each row's home, ``counts.shape[:-1]``.
        prior: Visit probability the rates shrink toward, broadcastable to ``counts``.
        strength: Smoothing strength ``a > 0``.

    Returns:
        Smoothed visit probabilities shaped like ``counts``.
    """
    rate = (counts + strength * prior) / (users.unsqueeze(-1) + strength)
    return rate.clamp(POPULARITY_EPS, 1 - POPULARITY_EPS)


def fit_smoothing_strength(
    counts: torch.Tensor,
    users: torch.Tensor,
    val: Users,
    val_prior: torch.Tensor,
    grid: torch.Tensor = SMOOTHING_GRID,
) -> float:
    """Pick the smoothing strength that maximises the Bernoulli log-likelihood of the validation users' non-home labels.

    Args:
        counts: Training visits indexed by (home, node), ``[n_nodes, n_nodes]``.
        users: Training residents per home, ``[n_nodes]``.
        val: Validation split.
        val_prior: Prior visit probability of each validation user and node, ``[n_val_users, n_nodes]``.
        grid: Candidate strengths.

    Returns:
        The best strength in ``grid``.
    """
    labels = val.visited.float()
    val_counts = counts[val.home_node]
    val_users = users[val.home_node]
    not_home = non_home_mask(val)

    log_likelihoods = []
    for strength in grid:
        p = smoothed_rate(val_counts, val_users, val_prior, strength)
        log_likelihood = labels * p.log() + (1 - labels) * (1 - p).log()
        log_likelihoods.append(log_likelihood[not_home].sum())

    return float(grid[torch.stack(log_likelihoods).argmax()])
