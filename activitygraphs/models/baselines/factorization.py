"""Matrix factorisation baseline: the MTF_BPR model of Phan et al. (2022), indexed by home node."""

import torch

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.models.scorers import Scorer

INIT_SCALE = 0.1


class HomeZoneMFBaseline(Scorer):
    """Factorisation of the home-zone-by-node visit matrix, trained by ``RankingModule``.

    ``s(h, n) = U_h . Z_n + b_n``, with ``h`` the user's home node. Phan's neighbourhood enrichment pools the visits of
    everyone living in a home zone, so the left factor is indexed by home node and both factors are
    ``[num_nodes, n_factors]``. The zone bias ``b_n`` is free. The home factors start at zero, so a home with no
    training resident keeps a zero factor and scores the bias alone.

    The scores are inner products rather than probabilities, so its BCE is not meaningful.

    Args:
        num_nodes: Number of nodes in the graph.
        n_factors: Latent dimension ``k``.

    Attributes:
        home_factors: ``U``, ``[num_nodes, n_factors]``.
        node_factors: ``Z``, ``[num_nodes, n_factors]``.
        bias: ``b``, ``[num_nodes]``.
    """

    def __init__(self, num_nodes: int, n_factors: int):
        super().__init__()
        self.home_factors = torch.nn.Parameter(torch.zeros(num_nodes, n_factors))
        self.node_factors = torch.nn.Parameter(torch.randn(num_nodes, n_factors) * INIT_SCALE)
        self.bias = torch.nn.Parameter(torch.zeros(num_nodes))

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the home-by-node score of every node for every user, ``[num_users, num_nodes]``."""
        home_factors = self.home_factors[batch.home_node]
        return home_factors @ self.node_factors.T + self.bias
