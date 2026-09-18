"""Computation of the home-excluded node popularity: i.e. the training visit rate of each node among the users who do
not live there."""

import torch

POPULARITY_EPS = 1e-6


def home_excluded_visit_rate(labels: torch.Tensor, home_idx: torch.Tensor) -> torch.Tensor:
    """Compute the visit rate of each node, over the users who do not live there.

    Args:
        labels: Binary visit labels, ``[n_users, n_nodes]``.
        home_idx: Home node of each user, ``[n_users]``.

    Returns:
        ``p_n``, ``[n_nodes]``: visits to ``n`` by non-residents of ``n`` over the number of non-residents, clamped to
        ``[1e-6, 1 - 1e-6]``.
    """
    n_users, n_nodes = labels.shape
    labels = labels.float()

    residents = torch.bincount(home_idx, minlength=n_nodes).float()
    resident_visits = torch.zeros(n_nodes, device=labels.device).index_add_(
        0, home_idx, labels[torch.arange(n_users, device=labels.device), home_idx]
    )

    visits = labels.sum(dim=0) - resident_visits
    candidates = (n_users - residents).clamp(min=1)

    return (visits / candidates).clamp(POPULARITY_EPS, 1 - POPULARITY_EPS)


def popularity_logit(labels: torch.Tensor, home_idx: torch.Tensor) -> torch.Tensor:
    """Compute ``logit(p_n)`` of ``home_excluded_visit_rate``.

    Args:
        labels: Binary visit labels, ``[n_users, n_nodes]``.
        home_idx: Home node of each user, ``[n_users]``.

    Returns:
        ``[n_nodes]`` popularity logits.
    """
    return torch.logit(home_excluded_visit_rate(labels, home_idx))


def home_visit_counts(labels: torch.Tensor, home_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the visits to each node by the residents of each home.

    Args:
        labels: Binary visit labels, ``[n_users, n_nodes]``.
        home_idx: Home node of each user, ``[n_users]``.

    Returns:
        ``(counts, users)``: ``[n_nodes, n_nodes]`` visits indexed by (home, node), and ``[n_nodes]`` residents per
        home.
    """
    n_nodes = labels.shape[1]
    counts = torch.zeros(n_nodes, n_nodes).index_add_(0, home_idx, labels.float())
    users = torch.bincount(home_idx, minlength=n_nodes).float()

    return counts, users
