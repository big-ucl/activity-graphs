"""Matrix factorisation baseline: the MTF_BPR model of Phan et al. (2022), indexed by home node."""

from typing import Self

import torch

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.models.baselines.common import non_home_mask
from activitygraphs.training.losses import dense_bpr_loss
from activitygraphs.training.metrics import per_user_recall
from activitygraphs.models.scorers import FittedScorer
from activitygraphs.models.baselines.popularity import home_visit_counts

N_FACTORS = 429
LEARNING_RATE = 1.91e-2
N_EPOCHS = 311
N_PAIRS = 128
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 10
INIT_SCALE = 0.1


class HomeZoneMFBaseline(FittedScorer):
    """Factorisation of the home-zone-by-node visit matrix, fitted by Bayesian personalised ranking.

    ``s(h, n) = U_h . Z_n + b_n``, with ``h`` the user's home node. Phan's neighbourhood enrichment replaces each
    user's row of the visit matrix by the pooled row of everyone living in their home zone, so the left factor is
    indexed by home node and both factors are ``[num_nodes, n_factors]``. The zone bias ``b_n`` is free. A home with
    no training resident keeps a zero factor and therefore scores the bias alone.

    The scores are inner products rather than probabilities, so its BCE is not meaningful.

    Args:
        num_nodes: Number of nodes in the graph.
        max_recall_k: Largest rank cutoff ``K`` of the validation ``avg_recall@K`` that selects the kept factors.
        n_factors: Latent dimension ``k``.
        lr: Adam learning rate.
        n_epochs: Full-batch epochs.
        n_pairs: Pairs sampled per home zone per epoch.
        weight_decay: L2 penalty on the factors and the bias.
        eval_every: Epochs between validation evaluations.
        seed: Seed of the factor initialisation and the pair sampling.

    Attributes:
        scores: ``[num_nodes, num_nodes]`` scores indexed by (home, node).
        val_avg_recall: Validation ``avg_recall@max_recall_k`` of the kept epoch.
        best_epoch: Epoch whose factors are kept.
    """

    scores: torch.Tensor
    val_avg_recall: float
    best_epoch: int

    def __init__(
        self,
        num_nodes: int,
        max_recall_k: int,
        n_factors: int = N_FACTORS,
        lr: float = LEARNING_RATE,
        n_epochs: int = N_EPOCHS,
        n_pairs: int = N_PAIRS,
        weight_decay: float = WEIGHT_DECAY,
        eval_every: int = EVAL_EVERY,
        seed: int = 0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_recall_k = max_recall_k
        self.n_factors = n_factors
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_pairs = n_pairs
        self.weight_decay = weight_decay
        self.eval_every = eval_every
        self.seed = seed

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Fit the factors by BPR on the home-pooled visit counts of the ``train`` users, keeping the best epoch.

        The best epoch is the one whose ``avg_recall`` over the ``val`` users is highest.
        """
        counts, _ = home_visit_counts(train.visited, train.home_node)
        pos_weight = counts.fill_diagonal_(0)
        neg_weight = (pos_weight == 0).float().fill_diagonal_(0)

        generator = torch.Generator().manual_seed(self.seed)
        home_factors = torch.randn(self.num_nodes, self.n_factors, generator=generator) * INIT_SCALE
        home_factors[pos_weight.sum(dim=1) == 0] = 0.0
        node_factors = torch.randn(self.num_nodes, self.n_factors, generator=generator) * INIT_SCALE
        bias = torch.zeros(self.num_nodes)

        params = [home_factors.requires_grad_(), node_factors.requires_grad_(), bias.requires_grad_()]
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        best = -float("inf")
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss = dense_bpr_loss(home_factors @ node_factors.T + bias, pos_weight, neg_weight, self.n_pairs, generator)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.eval_every != 0 and epoch + 1 != self.n_epochs:
                continue

            with torch.no_grad():
                scores = home_factors @ node_factors.T + bias

            recall = self._val_avg_recall(scores, val)
            if recall > best:
                best, self.best_epoch, self.scores = recall, epoch, scores

        self.val_avg_recall = best

        return self

    def _val_avg_recall(self, scores: torch.Tensor, val: Users) -> float:
        """Mean ``avg_recall@max_recall_k`` of the validation users, ranking every node but their own home."""
        user_scores = scores[val.home_node]
        candidates = non_home_mask(val)
        users = torch.arange(len(val.home_node)).unsqueeze(1).expand_as(user_scores)

        avg_recall, _, _ = per_user_recall(
            user_scores[candidates],
            val.visited[candidates],
            users[candidates],
            self.max_recall_k,
            [self.max_recall_k],
        )

        return float(avg_recall.mean())

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the fitted home-by-node score of every node for every user, ``[num_users, num_nodes]``."""
        return self.scores.to(batch.visited.device)[batch.home_node]
