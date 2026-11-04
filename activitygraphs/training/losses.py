"""Losses for per-user node-score: BCE and Pairwise ranking (BPR)."""

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as F

from activitygraphs.config import LossConfig
from activitygraphs.datasets.batch import UserBatch


class LossFn(Protocol):
    """Signature of a loss function over a batch's logits."""

    def __call__(
        self, logits: torch.Tensor, batch: UserBatch, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Return the scalar loss for ``logits`` against the labels in ``batch``."""


@dataclass(frozen=True)
class Loss:
    """A loss function with the validation metric and direction that checkpointing monitors."""

    name: str
    loss_fn: LossFn
    monitor: str
    monitor_mode: Literal["min", "max"]


def bce_loss(
    logits: torch.Tensor,
    batch: UserBatch,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Standard BCE loss with logits, over the ``[num_users, num_nodes]`` grid of scores."""
    return F.binary_cross_entropy_with_logits(logits, batch.visited.float())


BCELoss = Loss(name="bce", loss_fn=bce_loss, monitor="val_bce", monitor_mode="min")


def build_loss(cfg: LossConfig, max_recall_k: int) -> Loss:
    """Build the loss selected by ``cfg.type``, over the ``[num_users, num_nodes]`` grid of a ``UserBatch``."""
    if cfg.type == "bce":
        return BCELoss
    if cfg.type == "bpr":
        return _build_bpr_loss(cfg, max_recall_k)

    raise ValueError(f"Unknown loss type {cfg.type}")


def _build_bpr_loss(cfg: LossConfig, max_recall_k: int) -> Loss:
    def _uniform_bpr_loss(
        logits: torch.Tensor, batch: UserBatch, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return bpr_loss(logits, batch.visited, cfg.n_pairs, generator=generator)

    def _hard_neg_bpr_loss(
        logits: torch.Tensor, batch: UserBatch, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        near_home = batch.home_hops <= cfg.is_near_threshold
        return bpr_loss(logits, batch.visited, cfg.n_pairs, near_home, generator)

    _bpr_loss = _hard_neg_bpr_loss if cfg.neg_sampler == "hard" else _uniform_bpr_loss
    return Loss(name="bpr", loss_fn=_bpr_loss, monitor=f"val_avg_recall@{max_recall_k}", monitor_mode="max")


def bpr_loss(
    scores: torch.Tensor,
    y: torch.Tensor,
    n_pairs: int,
    neg_restrict: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean BPR loss over ``n_pairs`` sampled (v+, v-) pairs per user, over a grid of per-user scores.

    Args:
        scores: Score of every node for every user, ``[num_users, num_nodes]``.
        y: Visit labels aligned with ``scores``.
        n_pairs: Pairs drawn per user.
        neg_restrict: Boolean mask aligned with ``scores`` of the nodes a negative may be drawn from, e.g. near home
            for hard negatives. A user whose restricted pool is empty draws from every unvisited node.
        generator: Generator of the pair sampling.

    Returns:
        Scalar loss. Users with no positives are dropped.
    """
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    if neg_restrict is not None:
        neg_mask = _restrict_negatives(neg_mask, neg_restrict)

    return dense_bpr_loss(scores, pos_mask.float(), neg_mask.float(), n_pairs, generator)


def _restrict_negatives(neg_mask: torch.Tensor, neg_restrict: torch.Tensor) -> torch.Tensor:
    """Return the ``[B, N]`` mask ``neg_mask`` restricted to ``neg_restrict``.

    A user whose restricted pool is empty keeps every node of ``neg_mask``.
    """
    restricted = neg_mask & neg_restrict
    empty = ~restricted.any(dim=1, keepdim=True)

    return torch.where(empty, neg_mask, restricted)


def dense_bpr_loss(
    scores: torch.Tensor,
    pos_weight: torch.Tensor,
    neg_weight: torch.Tensor,
    n_pairs: int = 128,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean BPR loss over ``n_pairs`` (positive, negative) pairs sampled per row of a dense score matrix.

    Args:
        scores: Candidate scores, ``[n_rows, n_candidates]``.
        pos_weight: Non-negative weight of drawing each candidate as the positive of its row, ``[n_rows,
            n_candidates]``; a zero excludes the candidate.
        neg_weight: The same for the negative of the pair.
        n_pairs: Pairs drawn per row, so rows are weighted equally regardless of how many positives they hold.
        generator: Generator of the pair sampling.

    Returns:
        Scalar loss. Rows without both a positive and a negative are dropped; when no row has both, the loss is zero
        and still attached to ``scores``.
    """
    has_pairs = (pos_weight > 0).any(dim=1) & (neg_weight > 0).any(dim=1)
    if not has_pairs.any():
        return scores.sum() * 0.0  # keep Autodiff graph connected, zero-out loss

    row_scores = scores[has_pairs]
    pos_idx = torch.multinomial(pos_weight[has_pairs], n_pairs, replacement=True, generator=generator)
    neg_idx = torch.multinomial(neg_weight[has_pairs], n_pairs, replacement=True, generator=generator)

    diff = row_scores.gather(1, pos_idx) - row_scores.gather(1, neg_idx)  # [n_rows_with_pairs, n_pairs]
    return -F.logsigmoid(diff).mean()
