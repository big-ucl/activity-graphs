"""Losses for per-user node-score: BCE and Pairwise ranking (BPR)."""

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import to_dense_batch

from activitygraphs.config import LossConfig
from activitygraphs.ml.dataset import is_home_node_mask


class LossFn(Protocol):
    def __call__(
        self, logits: torch.Tensor, batch: pyg.data.Batch, generator: torch.Generator | None = None
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class Loss:
    name: str
    loss_fn: LossFn
    monitor: str
    monitor_mode: Literal["min", "max"]


def bce_loss(
    logits: torch.Tensor,
    batch: pyg.data.Batch,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Standard BCE loss with logits."""
    return F.binary_cross_entropy_with_logits(logits, batch.y.float())


BCELoss = Loss(name="bce", loss_fn=bce_loss, monitor="val_bce", monitor_mode="min")


def build_loss(cfg: LossConfig, distance_matrix: torch.Tensor, is_home_col_index: int, max_recall_k: int) -> Loss:
    if cfg.type == "bce":
        return BCELoss
    if cfg.type == "bpr":
        return _build_bpr_loss(cfg, distance_matrix, is_home_col_index, max_recall_k)

    raise ValueError(f"Unknown loss type {cfg.type}")


def _build_bpr_loss(cfg: LossConfig, distance_matrix: torch.Tensor, is_home_col_idx: int, max_recall_k: int) -> Loss:
    def _uniform_bpr_loss(logits: torch.Tensor, batch: pyg.data.Batch, generator: torch.Generator | None = None):
        return bpr_loss(logits, batch.y, batch.batch, n_pairs=cfg.n_pairs, generator=generator)

    def _hard_neg_bpr_loss(
        logits: torch.Tensor, batch: pyg.data.Batch, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        neg_restrict = _near_home_negative_mask(batch, distance_matrix, is_home_col_idx, cfg.is_near_threshold)
        return bpr_loss(logits, batch.y, batch.batch, cfg.n_pairs, neg_restrict, generator)

    _bpr_loss = _hard_neg_bpr_loss if cfg.neg_sampler == "hard" else _uniform_bpr_loss
    return Loss(name="bpr", loss_fn=_bpr_loss, monitor=f"val_avg_recall@{max_recall_k}", monitor_mode="max")


def bpr_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    n_pairs: int = 128,
    neg_restrict: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean BPR loss over ``n_pairs`` sampled (v+, v-) pairs per user.

    ``logits`` and ``y`` are ``[num_nodes, 1]`` (or ``[num_nodes]``); ``batch_index`` is
    ``batch.batch``. Negatives are this user's unvisited nodes, optionally restricted to
    ``neg_restrict`` (a ``[B, N]`` boolean mask, e.g. near-home for hard negatives). Each user
    contributes exactly ``n_pairs`` samples, so users are weighted equally regardless of |RG_i|.
     Users with no positives are dropped.
    """
    scores, mask = to_dense_batch(logits.reshape(-1), batch_index)  # [B, N], [B, N]
    labels, _ = to_dense_batch(y.reshape(-1).float(), batch_index)

    pos_mask = (labels > 0.5) & mask
    neg_mask = (labels <= 0.5) & mask
    if neg_restrict is not None:
        restricted = neg_mask & neg_restrict

        empty = ~restricted.any(
            dim=1, keepdim=True
        )  # fall back to all unvisited for users whose restricted pool is empty
        neg_mask = torch.where(empty, neg_mask, restricted)

    return dense_bpr_loss(scores, pos_mask.float(), neg_mask.float(), n_pairs, generator)


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


def _near_home_negative_mask(
    batch: pyg.data.Batch, distance_matrix: torch.Tensor, is_home_idx: int, is_near_threshold: int
) -> torch.Tensor:
    """Boolean ``[B, N]`` mask of unvisited nodes within ``hard_k`` hops of each user's home.

    Built from the same home-index logic as ``compute_home_hops``. Rows whose near-home
    pool is empty are left all-False; the caller falls back to all unvisited for those users.
    """
    device = batch.x.device
    distance_matrix = distance_matrix.to(device)

    node_idx = torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]
    is_home = is_home_node_mask(batch.x, is_home_idx)

    home_idx_per_graph = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
    home_idx_per_graph[batch.batch[is_home]] = node_idx[is_home]

    node_distance_to_home = distance_matrix[home_idx_per_graph[batch.batch], node_idx]  # [num_nodes]
    is_node_near_home = node_distance_to_home <= is_near_threshold

    is_node_near_home_dense, _ = to_dense_batch(is_node_near_home.float(), batch.batch)  # [B, N]
    return is_node_near_home_dense > 0.5
