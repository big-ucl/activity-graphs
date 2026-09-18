"""Tie-aware ranking statistics shared by the in-module metrics and the post-hoc analysis.

``positive_rank_stats`` and ``positive_band_auc`` work on torch tensors. ``expected_recall_at_k`` and ``average_recall`` only use arithmetic and
``clip``, so they accept torch tensors and numpy arrays alike.
"""

from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", torch.Tensor, np.ndarray)


def positive_rank_stats(
    scores: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count, for each positive, the candidates of its row that score higher than it and that tie with it.

    Args:
        scores: Candidate scores, ``[n_nodes]`` for one user or ``[n_users, n_nodes]``.
        target: Binary labels aligned with ``scores``.
        valid: Boolean mask aligned with ``scores`` of the entries that are candidates. All entries when None.

    Returns:
        ``(n_higher, n_tied)``, each ``[n_positives]``, one entry per valid positive in row-major order. ``n_tied``
        does not count the positive itself.
    """
    if scores.dim() == 1:
        scores, target = scores.unsqueeze(0), target.unsqueeze(0)
        valid = None if valid is None else valid.unsqueeze(0)

    valid = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid.bool()
    rows, cols = (target.bool() & valid).nonzero(as_tuple=True)

    positive_scores = scores[rows, cols].unsqueeze(1)
    row_scores = scores[rows]
    row_valid = valid[rows]

    n_higher = ((row_scores > positive_scores) & row_valid).sum(dim=1)
    n_tied = ((row_scores == positive_scores) & row_valid).sum(dim=1) - 1

    return n_higher, n_tied


def positive_band_auc(
    scores: torch.Tensor, target: torch.Tensor, bands: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each positive against the unvisited candidates of its own band.

    Args:
        scores: Candidate scores, ``[n_nodes]`` for one user or ``[n_users, n_nodes]``.
        target: Binary labels aligned with ``scores``.
        bands: Band index of each candidate, aligned with ``scores``.
        valid: Boolean mask aligned with ``scores`` of the entries that are candidates.

    Returns:
        ``(auc, n_negatives)``, each ``[n_positives]``, one entry per valid positive in row-major order. ``auc`` is
        the share of its band's unvisited candidates the positive outranks, counting each tie as half, and NaN where
        the band holds none of them. ``n_negatives`` is the number of unvisited candidates in the positive's band.
    """
    if scores.dim() == 1:
        scores, target = scores.unsqueeze(0), target.unsqueeze(0)
        bands, valid = bands.unsqueeze(0), valid.unsqueeze(0)

    valid = valid.bool()
    rows, cols = (target.bool() & valid).nonzero(as_tuple=True)

    positive_scores = scores[rows, cols].unsqueeze(1)
    positive_bands = bands[rows, cols].unsqueeze(1)

    row_scores = scores[rows]
    in_band_negative = ~target[rows].bool() & valid[rows] & (bands[rows] == positive_bands)

    n_negatives = in_band_negative.sum(dim=1)
    n_higher = ((row_scores > positive_scores) & in_band_negative).sum(dim=1)
    n_tied = ((row_scores == positive_scores) & in_band_negative).sum(dim=1)

    auc = 1.0 - (n_higher + n_tied / 2) / n_negatives

    return auc, n_negatives


def expected_recall_at_k(n_higher: Array, n_tied: Array, k: int | Array) -> Array:
    """Compute the tie-aware top-k hit fraction of each positive.

    Args:
        n_higher: Number of candidates scored higher than each positive.
        n_tied: Number of other candidates tied with each positive.
        k: Rank cutoff, broadcast against ``n_higher``.

    Returns:
        ``clip((k - n_higher) / (n_tied + 1), 0, 1)`` per positive: the fraction of the ranks the positive can take
        among its ties that are ``<= k``.
    """
    return ((k - n_higher) / (n_tied + 1)).clip(0, 1)


def average_recall(n_higher: Array, n_tied: Array, budget: int) -> Array:
    """Compute the mean of ``expected_recall_at_k`` over ``k = 1..budget`` for each positive.

    Args:
        n_higher: Number of candidates scored higher than each positive.
        n_tied: Number of other candidates tied with each positive.
        budget: Largest rank cutoff ``K``.

    Returns:
        Per-positive contribution to ``avg_recall@K``: 1 at rank 1, ``1 / K`` at rank ``K``, 0 beyond ``K``.
    """
    ramp_width = n_tied + 1
    n_cutoffs_past = (budget - n_higher).clip(0, None)
    n_cutoffs_in_ramp = ramp_width - (ramp_width - n_cutoffs_past).clip(0, None)
    n_cutoffs_after_ramp = (n_cutoffs_past - ramp_width).clip(0, None)

    ramp_sum = n_cutoffs_in_ramp * (n_cutoffs_in_ramp + 1) / (2 * ramp_width)

    return (ramp_sum + n_cutoffs_after_ramp) / budget
