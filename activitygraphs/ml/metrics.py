"""Diagnostic metrics: contiguity-hop distance and distance-from-home hop-band ranking metrics.

Aggregate ranking metrics are dominated by near-home positives. To detect whether a model
loses predictive power beyond its message-passing receptive field, ``ActivityGraphModule``
groups each node by its contiguity-hop distance from the user's home node into hop bands and
reports recall@k / ndcg@k within each band, using the per-hop-band torchmetrics collections
built here. Per-step ranking metrics use ``torchmetrics`` directly in the Lightning module;
there is intentionally no hand-rolled metric implementation.
"""

from collections.abc import Collection

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torchmetrics import MetricCollection, Metric
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall

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


def build_hop_band_metrics(hop_bands: Collection[tuple[str, float, float]], k: int) -> torch.nn.ModuleDict:
    """Build one ``recall@k`` / ``ndcg@k`` torchmetrics collection per hop band.

    Each collection is keyed by its hop-band label and prefixed ``test_hop_<label>_`` so the
    logged keys (e.g. ``test_hop_3-5_recall@5``) are captured by the test results row.
    """
    return torch.nn.ModuleDict({
        label: MetricCollection(
            {
                f"recall@{k}": RetrievalRecall(top_k=k, empty_target_action="skip"),
                f"ndcg@{k}": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            },
            prefix=f"test_hop_{label}_",
        )
        for label, _, _ in hop_bands
    })


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
