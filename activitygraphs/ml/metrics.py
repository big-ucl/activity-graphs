"""Diagnostic metrics: contiguity-hop distance and distance-from-home hop-band ranking metrics.

Aggregate ranking metrics are dominated by near-home positives. To detect whether a model
loses predictive power beyond its message-passing receptive field, ``ActivityGraphModule``
groups each node by its spatial adjacency-hop distance from the user's home node into hop bands.
It reports recall@k / ndcg@k within each band, using the per-hop-band torchmetrics collections
built here. Regular per-step ranking metrics use ``torchmetrics`` directly in the Lightning module.
"""

from collections.abc import Collection

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torchmetrics import MeanMetric, Metric, MetricCollection, SumMetric
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall
from torchmetrics.utilities.data import dim_zero_cat

# Hop bands: (label, low_hops_inclusive, high_hops_inclusive)
DEFAULT_HOP_BANDS: tuple[tuple[str, float, float], ...] = (
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


class HopBandMetrics(torch.nn.Module):
    """Ranking, NLL and positive-count metrics computed separately within each hop band.

    Ranking is restricted to the nodes inside a band, so each band answers "within this distance
    ring, does the model separate visited from unvisited nodes" - the distance decay a model gets
    for free from home is factored out. ``compute`` returns one row per band in long format, which
    both the scalar logging keys and the W&B ``hop_bands`` table are derived from.
    """

    def __init__(self, hop_bands: Collection[tuple[str, float, float]], k: int) -> None:
        super().__init__()
        self.hop_bands = tuple(hop_bands)
        self.k = k
        self.ranking = torch.nn.ModuleDict({
            label: MetricCollection({
                "recall": RetrievalRecall(top_k=k, empty_target_action="skip"),
                "ndcg": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            })
            for label, _, _ in self.hop_bands
        })
        self.nll = torch.nn.ModuleDict({label: MeanMetric() for label, _, _ in self.hop_bands})
        self.n_pos = torch.nn.ModuleDict({label: SumMetric() for label, _, _ in self.hop_bands})
        self._seen: set[str] = set()

    def update(self, hops: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, user_ids: torch.Tensor) -> None:
        """Accumulate one batch, given each node's hop distance from its user's home node."""
        scores = logits.sigmoid()

        for label, low, high in self.hop_bands:
            mask = (hops >= low) & (hops <= high)
            if not mask.any():
                continue

            band_target = target[mask]
            self.ranking[label].update(scores[mask], band_target, indexes=user_ids[mask])
            self.nll[label].update(
                F.binary_cross_entropy_with_logits(logits[mask], band_target.float(), reduction="none")
            )
            self.n_pos[label].update(band_target.sum())

            if band_target.sum() > 0:
                self._seen.add(label)

    def compute(self) -> list[dict[str, float | str]]:
        """Return one row per band, in band order, skipping bands that saw no positives.

        Bands with no positives are skipped. The lower bound of the hop band, ``hop_low``, is
        included as a numeric sort key.
        """
        rows: list[dict[str, float | str]] = []

        for label, low, _ in self.hop_bands:
            if label not in self._seen:
                continue

            row: dict[str, float | str] = {"hop_band": label, "hop_low": low, "k": self.k}
            row |= {name: value.item() for name, value in self.ranking[label].compute().items()}
            row |= {"nll": self.nll[label].compute().item(), "n_pos": self.n_pos[label].compute().item()}
            rows.append(row)

        return rows

    def reset(self) -> None:
        for label, _, _ in self.hop_bands:
            self.ranking[label].reset()
            self.nll[label].reset()
            self.n_pos[label].reset()
        self._seen.clear()


def hop_band_scalars(rows: Collection[dict[str, float | str]]) -> dict[str, float]:
    """Flatten hop-band rows into ``test_hop_<label>_<metric>`` scalar logging keys.

    Ranking metrics keep their ``@k`` suffix (e.g. ``test_hop_3-5_recall@25``); ``nll`` and
    ``n_pos`` do not, since they are not cutoff-dependent.
    """
    scalars = {}

    for row in rows:
        prefix = f"test_hop_{row['hop_band']}_"
        for name in ("recall", "ndcg"):
            scalars[f"{prefix}{name}@{row['k']}"] = float(row[name])
        for name in ("nll", "n_pos"):
            scalars[f"{prefix}{name}"] = float(row[name])

    return scalars


class RetrievalRPrecision(Metric):
    """Per-user recall at k = number of visited nodes (R-precision).

    For each user, scores the top-R nodes where R is that user's realised set size |RG_i|,
    and reports hits / R, averaged over users. The cutoff tracks each user's realised set
    size, so it stays comparable across datasets with very different set sizes (Geneva ~2-4,
    Toronto ~26) where a fixed @5 does not.
    """

    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        for idx in torch.unique(indexes):
            mask = indexes == idx
            t = target[mask]
            r = int(t.sum())
            if r == 0:
                continue
            top = preds[mask].topk(min(r, t.numel())).indices
            self.score_sum = self.score_sum + t[top].sum() / r
            self.n_users = self.n_users + 1

    def compute(self) -> torch.Tensor:
        return self.score_sum / self.n_users.clamp(min=1)


class PerUserRanking(Metric):
    """Per-user test-time ranking scores, retained rather than averaged.

    An aggregate ranking metric collapses the test split to a single number, which cannot
    separate a real difference between two models from run-to-run noise. Keeping the per-user
    score lets two models be compared on the same users with a paired test (bootstrap or
    signed-rank), which is far more sensitive than differencing two aggregates. ``n_pos`` is
    retained alongside so comparisons can be stratified by realised set size.
    """

    full_state_update = False

    user_id: list[torch.Tensor]
    n_pos: list[torch.Tensor]
    r_precision: list[torch.Tensor]
    recall: list[torch.Tensor]
    score_vector: list[torch.Tensor]

    _FIELDS = ("user_id", "n_pos", "r_precision", "recall")

    def __init__(self, k: int, store_score_vectors: bool = False) -> None:
        super().__init__()
        self.k = k
        self.store_score_vectors = store_score_vectors
        for field in self._FIELDS:
            self.add_state(field, default=[], dist_reduce_fx="cat")

        self.add_state("score_vector", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        """Accumulate one row per user in the batch, skipping users with no positives."""
        for idx in torch.unique(indexes):
            mask = indexes == idx
            t = target[mask]
            r = int(t.sum())
            if r == 0:
                continue

            scores = preds[mask]
            top_r = scores.topk(min(r, t.numel())).indices
            top_k = scores.topk(min(self.k, t.numel())).indices

            self.user_id.append(idx.reshape(1).long())
            self.n_pos.append(torch.full((1,), r, dtype=torch.long, device=scores.device))
            self.r_precision.append((t[top_r].sum() / r).reshape(1).float())
            self.recall.append((t[top_k].sum() / r).reshape(1).float())

            if self.store_score_vectors:
                self.score_vector.append(scores.detach().float().reshape(1, -1).cpu())

    def columns(self) -> dict[str, torch.Tensor]:
        """Return the retained per-user columns, each a 1-D tensor of length n_users.

        Use this rather than ``compute``: the torchmetrics wrapper around ``compute`` squeezes
        single-element outputs to 0-dim, which would turn a one-user split into a scalar.
        """
        if not self.user_id:
            return {field: torch.empty(0) for field in self._FIELDS}

        return {field: dim_zero_cat(getattr(self, field)) for field in self._FIELDS}

    def score_vectors(self) -> torch.Tensor:
        """Return the retained score vectors as ``[n_users, num_nodes]``, row-aligned with ``columns()``.

        Empty when ``store_score_vectors`` is off. The scores are the model's per-node ranking scores over
        the whole node universe, which the summary columns reduce away. The model-health checks are computed
        from them and cannot be recovered from the summaries afterwards.
        """
        if not self.score_vector:
            return torch.empty(0)

        return dim_zero_cat(self.score_vector)

    def compute(self) -> dict[str, torch.Tensor]:
        return self.columns()
