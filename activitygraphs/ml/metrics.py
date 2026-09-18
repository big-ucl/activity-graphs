"""Ranking metrics (average recall, R-precision, per-user ranking) and the contiguity-hop distance."""

from collections.abc import Collection, Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_dense_batch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from activitygraphs.ml.ranking import average_recall, expected_recall_at_k, positive_rank_stats

# Hop bands: (label, low_hops_inclusive, high_hops_inclusive)
HopBands = Collection[tuple[str, float, float]]

DEFAULT_HOP_BANDS: HopBands = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13+", 13, float("inf")),
)


def compute_home_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return the all-pairs contiguity-hop distance matrix ``[num_nodes, num_nodes]``.

    Treats the graph as undirected and unweighted; unreachable pairs are ``inf``.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))

    return shortest_path(adjacency, method="D", unweighted=True, directed=False)


@torch.no_grad()
def per_user_recall(
    preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, budget: int, ks: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute each user's ``avg_recall@budget`` and tie-aware ``recall@k``, over the users in a flat batch.

    Args:
        preds: Candidate scores, ``[n_rows]``.
        target: Binary labels aligned with ``preds``, ``[n_rows]``.
        indexes: User id of each row, ``[n_rows]``.
        budget: Largest rank cutoff ``K`` of ``avg_recall@K``.
        ks: Rank cutoffs of the returned ``recall@k``.

    Returns:
        ``(avg_recall, recall_at_ks, n_dropped)``: ``[n_scored_users]`` and ``[n_scored_users, len(ks)]`` for the
        users with at least one positive, and the number of users without one.
    """
    _, user_row = torch.unique(indexes, return_inverse=True)
    order = torch.argsort(user_row, stable=True)

    scores, valid = to_dense_batch(preds[order], user_row[order])
    labels, _ = to_dense_batch(target[order].bool(), user_row[order], fill_value=False)
    n_pos = (labels & valid).sum(dim=1)

    n_higher, n_tied = positive_rank_stats(scores, labels, valid)
    positive_row = torch.repeat_interleave(torch.arange(len(n_pos), device=preds.device), n_pos)
    cutoffs = torch.as_tensor(list(ks), device=preds.device)

    avg_recall_sum = torch.zeros(len(n_pos), device=preds.device).index_add_(
        0, positive_row, average_recall(n_higher, n_tied, budget).float()
    )
    recall_sum = torch.zeros(len(n_pos), len(cutoffs), device=preds.device).index_add_(
        0, positive_row, expected_recall_at_k(n_higher[:, None], n_tied[:, None], cutoffs[None, :]).float()
    )

    scored = n_pos > 0
    denominator = n_pos[scored].float()

    return avg_recall_sum[scored] / denominator, recall_sum[scored] / denominator[:, None], int((~scored).sum())


class RetrievalAverageRecall(Metric):
    """Per-user mean recall@k over ``k = 1..budget``, plus the tie-aware recall@k at ``ks``, averaged over users.

    Users without a positive are dropped from the average and counted in ``n_dropped``.

    Args:
        budget: Largest rank cutoff ``K`` of ``avg_recall@K``.
        ks: Rank cutoffs of the reported ``recall@k``.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, budget: int, ks: Sequence[int]) -> None:
        super().__init__()
        self.budget = budget
        self.ks = list(ks)
        self.add_state("avg_recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall_sum", default=torch.zeros(len(self.ks)), dist_reduce_fx="sum")
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_dropped", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        avg_recall, recall_at_ks, n_dropped = per_user_recall(preds, target, indexes, self.budget, self.ks)

        self.avg_recall_sum = self.avg_recall_sum + avg_recall.sum()
        self.recall_sum = self.recall_sum + recall_at_ks.sum(dim=0)
        self.n_users = self.n_users + len(avg_recall)
        self.n_dropped = self.n_dropped + n_dropped

    def compute(self) -> dict[str, torch.Tensor]:
        """Returns ``avg_recall@{budget}`` and ``recall@{k}`` for each ``k`` in ``ks``."""
        n_users = self.n_users.clamp(min=1)
        recall_at_ks = self.recall_sum / n_users

        return {
            f"avg_recall@{self.budget}": self.avg_recall_sum / n_users,
            **{f"recall@{k}": recall_at_ks[i] for i, k in enumerate(self.ks)},
        }

    def user_counts(self) -> dict[str, float]:
        """Returns the count of users that were scored and the count of users that were dropped (empty ``RG_i``)."""
        return {"n_scored_users": float(self.n_users), "n_dropped_users": float(self.n_dropped)}


class RetrievalRPrecision(Metric):
    """Per-user recall at k = number of visited nodes (R-precision).

    For each user, scores the top-R nodes where R is that user's realised set size |RG_i|,
    and reports hits / R, averaged over users. Users that have no visits are dropped from the average (e.g. when only
    home node is visited and is dropped, masked out by the caller). ``n_dropped_users`` tracks the number of such users.

    R-precision is comparable across dataset since it compares over the realised set size.
    """

    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_dropped", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        for idx in torch.unique(indexes):
            mask = indexes == idx
            t = target[mask]
            r = int(t.sum())
            if r == 0:
                self.n_dropped = self.n_dropped + 1
                continue
            top = preds[mask].topk(min(r, t.numel())).indices
            self.score_sum = self.score_sum + t[top].sum() / r
            self.n_users = self.n_users + 1

    def compute(self) -> torch.Tensor:
        return self.score_sum / self.n_users.clamp(min=1)


class PerUserRanking(Metric):
    """Ranking scores for each test user, disaggregate.

    Keeps the ranking scores and positive counts of each test user for futher analyses, as a struct of lists, indexed
    by users:
        - n_pos: number of positives for that user
        - n_pos_home_incl: number of positives for that user, including the home node
        - r_precision: R-precision for this user

    Also keeps info about each of the users' positives (i.e. visited nodes, R = |RG_i|), NOT indexed by user, but by
    ``sum(num_pos_u, u in users)`` (i.e. each index is a visited node), struct of flattened lists:
        - pos_hops: number of hops from home to the positive node
        - pos_in_top_r: did this positive node make it into the top R (= |RG_i|) nodes, ties broken arbitrarily
        - pos_n_scored_higher: number of nodes ranked higher than the positive node
        - pos_n_tied: number of nodes that tied with the positive node
    """

    full_state_update = False

    user_id: list[torch.Tensor]
    n_pos: list[torch.Tensor]
    n_pos_home_incl: list[torch.Tensor]
    r_precision: list[torch.Tensor]

    pos_hops: list[torch.Tensor]
    pos_in_top_r: list[torch.Tensor]
    pos_n_scored_higher: list[torch.Tensor]
    pos_n_tied: list[torch.Tensor]

    score_vector: list[torch.Tensor]

    _FIELDS = ("user_id", "n_pos", "n_pos_home_incl", "r_precision")
    _POSITIVE_FIELDS = ("pos_hops", "pos_in_top_r", "pos_n_scored_higher", "pos_n_tied")

    def __init__(self, store_score_vectors: bool = False) -> None:
        super().__init__()
        self.store_score_vectors = store_score_vectors
        for field in self._FIELDS + self._POSITIVE_FIELDS:
            self.add_state(field, default=[], dist_reduce_fx="cat")

        self.add_state("score_vector", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        exclude: torch.Tensor,
        hops: torch.Tensor,
    ) -> None:
        """Update the per-user and per-positive tensors with the results from the current batch. Skips users with no
        positives.

        Args:
            preds: flat tensor of predicted scores, [n_users * n_nodes]
            target: tensor of binary visit labels, aligned with ``preds``, [n_users * n_nodes]
            indexes: tensor of user ids of each row, [n_users * n_nodes]
            exclude: boolean mask of the rows to be excluded from ranking (e.g. ``is_home`` flag), , aligned with
                ``preds``, [n_users * n_nodes]
            hops: tensor of hop distances for each node, aligned with ``preds``, [n_users * n_nodes]
        """
        for idx in torch.unique(indexes):
            mask = indexes == idx

            full_target = target[mask]
            full_scores = preds[mask]

            non_excluded_nodes = ~exclude[mask]
            scored_target = full_target[non_excluded_nodes]
            scored_scores = full_scores[non_excluded_nodes]
            scored_hops = hops[mask][non_excluded_nodes]

            num_positives = int(scored_target.sum())
            if num_positives == 0:
                continue

            top_r = scored_scores.topk(min(num_positives, scored_target.numel())).indices

            full_num_positives = full_target.sum()
            r_precision = scored_target[top_r].sum() / num_positives

            is_positive = scored_target.bool()
            in_top_r = torch.zeros_like(is_positive)
            in_top_r[top_r] = True

            n_scored_higher, n_tied = positive_rank_stats(scored_scores, scored_target)

            self.user_id.append(idx.reshape(1).long())
            self.n_pos.append(torch.full((1,), num_positives, dtype=torch.long, device=scored_scores.device))
            self.n_pos_home_incl.append(full_num_positives.reshape(1).long())
            self.r_precision.append(r_precision.reshape(1).float())

            self.pos_hops.append(scored_hops[is_positive].float())
            self.pos_in_top_r.append(in_top_r[is_positive])
            self.pos_n_scored_higher.append(n_scored_higher.long())
            self.pos_n_tied.append(n_tied.long())

            if self.store_score_vectors:
                self.score_vector.append(full_scores.detach().float().reshape(1, -1).cpu())

    def columns(self) -> dict[str, torch.Tensor]:
        """Return the per-user columns as a mapping (col_name -> tensor of values), values are indexed by users.  See
        class docstring for description of columns.

        Use this rather than ``compute``: the torchmetrics wrapper around ``compute`` squeezes
        single-element outputs to 0-dim, which would turn a one-user split into a scalar.
        """
        return {field: dim_zero_cat(getattr(self, field)) for field in self._FIELDS}

    def positive_columns(self) -> dict[str, list[list]]:
        """Return the positive (visited nodes) columns as a mapping (col_name -> tensor of values), values are indexed
        by positives (i.e. ``sum(num_pos_u for u in users)``). See class docstring for description of columns.
        """
        values = {pos_field: dim_zero_cat(getattr(self, pos_field)).tolist() for pos_field in self._POSITIVE_FIELDS}
        positives: dict[str, list[list]] = {pos_field: [] for pos_field in self._POSITIVE_FIELDS}

        start = 0
        for count in dim_zero_cat(self.n_pos).tolist():
            end = start + count
            for pos_field in self._POSITIVE_FIELDS:
                positives[pos_field].append(values[pos_field][start:end])
            start = end

        return positives

    def score_vectors(self) -> torch.Tensor:
        """Return the tensor of model score vectors, row-aligned with ``columns()``, ``[n_users, num_nodes]``. Empty
        when ``store_score_vectors`` is off."""
        if not self.score_vector:
            return torch.empty(0)

        return dim_zero_cat(self.score_vector)

    def compute(self) -> dict[str, torch.Tensor]:
        return self.columns()
