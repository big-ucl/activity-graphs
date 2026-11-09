"""Ranking metrics (average recall, R-precision, per-user ranking)."""

from collections.abc import Collection, Sequence

import torch
from torch_geometric.utils import to_dense_batch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from activitygraphs.training.ranking import average_recall, expected_recall_at_k, positive_rank_stats

# Hop bands: (label, low_hops_inclusive, high_hops_inclusive)
HopBands = Collection[tuple[str, float, float]]

DEFAULT_HOP_BANDS: HopBands = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13+", 13, float("inf")),
)


@torch.no_grad()
def per_user_recall(
    preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, budget: int, ks: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute each user's ``avg_recall@budget`` and tie-aware ``recall@k``, over the users in a flat batch.

    Args:
        preds: Candidate scores, ``[n_rows]``.
        target: Binary labels aligned with ``preds``, ``[n_rows]``.
        indexes: Dataset index of the user of each row, ``[n_rows]``.
        budget: Largest rank cutoff ``K`` of ``avg_recall@K``.
        ks: Rank cutoffs of the returned ``recall@k``.

    Returns:
        ``(avg_recall, recall_at_ks, n_dropped)``: ``[n_scored_users]`` and ``[n_scored_users, len(ks)]`` for the
        users with at least one positive, and the number of users without one.
    """
    labels = target.bool()
    _, (scores, labels), valid = _dense_by_user(indexes, (preds, labels))
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
        """Accumulate per-user average recall and recall@k over a batch."""
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
        """Accumulate per-user R-precision over a batch, counting users with no positives as dropped."""
        labels = target.bool()
        _, (scores, labels), valid = _dense_by_user(indexes, (preds, labels))
        n_pos, in_top_r = _top_r(scores, labels, valid)

        is_user_scored_mask = n_pos > 0
        hits = (in_top_r & labels).sum(dim=1)

        self.score_sum = self.score_sum + (hits[is_user_scored_mask] / n_pos[is_user_scored_mask]).sum()
        self.n_users = self.n_users + is_user_scored_mask.sum()
        self.n_dropped = self.n_dropped + (~is_user_scored_mask).sum()

    def compute(self) -> torch.Tensor:
        """Return the mean R-precision over scored users."""
        return self.score_sum / self.n_users.clamp(min=1)


class UserPairAccuracy(Metric):
    """Mean over users of the share of each user's (visited, unvisited) node pairs in which the visited node scores higher.

    Every node is a candidate, the home node included, as in the BPR loss. A tied pair counts as one half. Users without
    both a visited and an unvisited node are left out of the mean.
    """

    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, scores: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate the pair accuracy of every user of a batch.

        Args:
            scores: Score of every node for every user, ``[n_users, n_nodes]``.
            target: Binary labels aligned with ``scores``.
        """
        accuracy = _per_user_pair_accuracy(scores, target)

        self.score_sum = self.score_sum + accuracy.sum()
        self.n_users = self.n_users + accuracy.numel()

    def compute(self) -> torch.Tensor:
        """Return the mean pair accuracy over the users with pairs."""
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
        - pos_node: index of the positive node within the graph
        - pos_hops: number of hops from home to the positive node
        - pos_in_top_r: did this positive node make it into the top R (= |RG_i|) nodes, ties broken arbitrarily
        - pos_n_scored_higher: number of nodes ranked higher than the positive node
        - pos_n_tied: number of nodes that tied with the positive node

    Args:
        store_score_vectors: Keep each user's full per-node score vector.
    """

    full_state_update = False

    user_idx: list[torch.Tensor]
    n_pos: list[torch.Tensor]
    n_pos_home_incl: list[torch.Tensor]
    r_precision: list[torch.Tensor]

    pos_node: list[torch.Tensor]
    pos_hops: list[torch.Tensor]
    pos_in_top_r: list[torch.Tensor]
    pos_n_scored_higher: list[torch.Tensor]
    pos_n_tied: list[torch.Tensor]

    score_vector: list[torch.Tensor]

    _FIELDS = ("user_idx", "n_pos", "n_pos_home_incl", "r_precision")
    _POSITIVE_FIELDS = ("pos_node", "pos_hops", "pos_in_top_r", "pos_n_scored_higher", "pos_n_tied")

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
        """Update the per-user and per-positive tensors with the results from the current batch.

        Skips users with no positives.

        Args:
            preds: flat tensor of predicted scores, [n_users * n_nodes]
            target: tensor of binary visit labels, aligned with ``preds``, [n_users * n_nodes]
            indexes: tensor of the dataset index of the user of each row, [n_users * n_nodes]
            exclude: boolean mask of the rows to be excluded from ranking (e.g. ``is_home`` flag), , aligned with
                ``preds``, [n_users * n_nodes]
            hops: tensor of hop distances for each node, aligned with ``preds``, [n_users * n_nodes]
        """
        labels = target.bool()
        users, (scores, labels, excluded, hop_rows), valid = _dense_by_user(indexes, (preds, labels, exclude, hops))
        candidates = valid & ~excluded

        n_pos_home_incl = (labels & valid).sum(dim=1)
        n_pos, in_top_r = _top_r(scores, labels, candidates)

        kept = n_pos > 0
        scores = scores[kept]
        labels = labels[kept]
        candidates = candidates[kept]
        hop_rows = hop_rows[kept]
        in_top_r = in_top_r[kept]
        n_pos = n_pos[kept]

        r_precision = (in_top_r & labels).sum(dim=1) / n_pos
        n_scored_higher, n_tied = positive_rank_stats(scores, labels, candidates)
        rows, nodes = (labels & candidates).nonzero(as_tuple=True)

        self.user_idx.append(users[kept].long())
        self.n_pos.append(n_pos.long())
        self.n_pos_home_incl.append(n_pos_home_incl[kept].long())
        self.r_precision.append(r_precision.float())

        self.pos_node.append(nodes.long())
        self.pos_hops.append(hop_rows[rows, nodes].float())
        self.pos_in_top_r.append(in_top_r[rows, nodes])
        self.pos_n_scored_higher.append(n_scored_higher.long())
        self.pos_n_tied.append(n_tied.long())

        if self.store_score_vectors:
            self.score_vector.append(scores.detach().float().cpu())

    def columns(self) -> dict[str, torch.Tensor]:
        """Return the per-user columns as a mapping (col_name -> tensor of values), values are indexed by users.

        See class docstring for description of columns.

        Use this rather than ``compute``: the torchmetrics wrapper around ``compute`` squeezes
        single-element outputs to 0-dim, which would turn a one-user split into a scalar.
        """
        return {field: dim_zero_cat(getattr(self, field)) for field in self._FIELDS}

    def positive_columns(self) -> dict[str, list[list]]:
        """Return the positive (visited nodes) columns as a mapping (col_name -> tensor of values).

        Values are indexed by positives (i.e. ``sum(num_pos_u for u in users)``). See class docstring for description
        of columns.
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
        """Return the tensor of model score vectors, row-aligned with ``columns()``, ``[n_users, num_nodes]``.

        Empty when ``store_score_vectors`` is off.
        """
        if not self.score_vector:
            return torch.empty(0)

        return dim_zero_cat(self.score_vector)

    def compute(self) -> dict[str, torch.Tensor]:
        """Return the per-user ranking columns."""
        return self.columns()


def _dense_by_user(
    indexes: torch.Tensor, values: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Convert flat rows into one row per user, keeping the order within each user's rows.

    Args:
        indexes: Dataset index of the user of each flat row, ``[n_rows]``.
        values: List of tensors aligned with ``indexes``, each ``[n_rows]``.

    Returns:
        A tuple of ``(users, dense, valid)``:
            - ``users``: the distinct dataset indices in ascending order, ``[n_users]``,
            - each tensor of ``values``, re-formed as ``[n_users, max_rows_per_user]``, and
            - the mask of the entries that hold a row, ``[n_users, max_rows_per_user]``. The entries outside the mask
               hold arbitrary values.
    """
    users, user_row = torch.unique(indexes, return_inverse=True)
    order = torch.argsort(user_row, stable=True)
    row_of_entry, valid = to_dense_batch(order, user_row[order])

    return users, [value[row_of_entry] for value in values], valid


def _top_r(scores: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify each user's ``R`` highest-scored candidates, ``R`` being the number of positives among the candidates.

    Among tied candidates, the ones earlier in the row are placed first.

    Args:
        scores: Candidate scores, ``[n_users, n_nodes]``.
        target: Boolean labels aligned with ``scores``.
        valid: Boolean mask aligned with ``scores`` of the entries that are candidates.

    Returns:
        ``(n_pos, in_top_r)``: the number of positive candidates of each user, ``[n_users]``, and the mask of the
        candidates in the user's top ``R``, ``[n_users, n_nodes]``.
    """
    n_pos = (target & valid).sum(dim=1)

    masked_scores = scores.masked_fill(~valid, float("-inf"))
    order = masked_scores.argsort(dim=1, descending=True, stable=True)
    positions = torch.arange(scores.shape[1], device=scores.device).expand_as(order)
    rank = torch.empty_like(order).scatter_(1, order, positions)

    in_top_r = (rank < n_pos[:, None]) & valid

    return n_pos, in_top_r


@torch.no_grad()
def _per_user_pair_accuracy(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute each user's share of (visited, unvisited) node pairs in which the visited node scores higher.

    Every node is a candidate, the home node included, as in the BPR loss. A tied pair counts as one half.

    Args:
        scores: Score of every node for every user, ``[n_users, n_nodes]``.
        target: Binary labels aligned with ``scores``.

    Returns:
        ``[n_users_with_pairs]``, for the users with at least one visited and one unvisited node.
    """
    positive = target.bool()
    negative = ~positive
    rows, cols = positive.nonzero(as_tuple=True)

    positive_scores = scores[rows, cols].unsqueeze(1)
    row_scores = scores[rows]
    row_negative = negative[rows]

    n_lower = ((row_scores < positive_scores) & row_negative).sum(dim=1)
    n_tied = ((row_scores == positive_scores) & row_negative).sum(dim=1)
    n_ordered = n_lower.float() + 0.5 * n_tied.float()
    n_ordered_per_user = torch.zeros(len(scores), device=scores.device).index_add_(0, rows, n_ordered)

    n_pairs = positive.sum(dim=1) * negative.sum(dim=1)
    has_pairs = n_pairs > 0

    return n_ordered_per_user[has_pairs] / n_pairs[has_pairs].float()
