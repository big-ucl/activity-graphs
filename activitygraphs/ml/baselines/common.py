"""Common utilies for baseline models."""

from dataclasses import dataclass

import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_batch

from activitygraphs.ml.dataset import is_home_node_mask
from activitygraphs.ml.popularity import POPULARITY_EPS


@dataclass(frozen=True)
class PerUserTensors:
    """Contains the per-user tensors of a dataset (in dense format, indexed by users).

    Attributes:
        labels: Binary visit labels, ``[n_users, n_nodes]``.
        home_idx: Home node of each user, ``[n_users]``.
        distances: Distance from each user's home to every node, as stored in the batches (scaled),
            ``[n_users, n_nodes]``.
    """

    labels: torch.Tensor
    home_idx: torch.Tensor
    distances: torch.Tensor


def extract_per_user_tensors(loader: pyg.loader.DataLoader, is_home_idx: int) -> PerUserTensors:
    """Takes a dataloader and extracts the per-user information into dense CPU tensors.

    Args:
        loader: DataLoader over a dataset; every graph must have the same node count.
        is_home_idx: Column index of ``is_home`` in ``batch.x``.

    Returns:
        The dataloader's ``SplitTensors``.

    Raises:
        ValueError: If a graph has no home node.
    """
    labels, homes, distances = [], [], []

    for batch in loader:
        batch = batch.to("cpu")
        is_home, _ = to_dense_batch(is_home_node_mask(batch.x, is_home_idx), batch.batch, fill_value=False)
        if not is_home.any(dim=1).all():
            raise ValueError("No home node found.")

        labels.append(to_dense_batch(batch.y.reshape(-1), batch.batch)[0])
        homes.append(is_home.int().argmax(dim=1))
        distances.append(to_dense_batch(batch.distances.reshape(-1), batch.batch)[0])

    return PerUserTensors(labels=torch.cat(labels), home_idx=torch.cat(homes), distances=torch.cat(distances))


def non_home_mask(per_user_tensors: PerUserTensors) -> torch.Tensor:
    """Return the ``[n_users, n_nodes]`` mask of the (user, node) pairs where the node is not the user's home."""
    mask = torch.ones_like(per_user_tensors.labels, dtype=torch.bool)
    mask[torch.arange(len(mask)), per_user_tensors.home_idx] = False
    return mask


def graph_node_indices(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Return, for each node, its within-graph index, ``[num_nodes]``."""
    ptr = torch.zeros(batch.max().item() + 2, dtype=torch.long, device=batch.device)
    ptr[1:] = batch.bincount().cumsum(0)
    return torch.arange(x.shape[0], device=x.device) - ptr[batch]


def graph_home_indices(x: torch.Tensor, batch: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Return, for each node, the within-graph index of the corresponding home node, ``[num_nodes]``.

    Raises:
        ValueError: If a graph has no home node.
    """
    is_home, _ = to_dense_batch(is_home_node_mask(x, is_home_idx), batch, fill_value=False)
    if not is_home.any(dim=1).all():
        raise ValueError("No home node found.")

    return is_home.int().argmax(dim=1)[batch]


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
    val: PerUserTensors,
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
    labels = val.labels.float()
    val_counts = counts[val.home_idx]
    val_users = users[val.home_idx]
    not_home = torch.ones_like(labels, dtype=torch.bool)
    not_home[torch.arange(len(labels)), val.home_idx] = False

    log_likelihoods = []
    for strength in grid:
        p = smoothed_rate(val_counts, val_users, val_prior, strength)
        log_likelihood = labels * p.log() + (1 - labels) * (1 - p).log()
        log_likelihoods.append(log_likelihood[not_home].sum())

    return float(grid[torch.stack(log_likelihoods).argmax()])
