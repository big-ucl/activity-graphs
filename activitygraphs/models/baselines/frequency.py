"""Frequency-based (no learning) baselines for visit prediction."""

from typing import Self

import torch

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.models.baselines.common import fit_smoothing_strength, smoothed_rate
from activitygraphs.models.scorers import FittedScorer
from activitygraphs.models.baselines.popularity import home_excluded_visit_rate, home_visit_counts, popularity_logit


def inverse_sigmoid(prob):
    """Return logit(prob) = log(prob / (1 - prob))."""
    return torch.log(prob / (1 - prob))


class UniformBaseline(FittedScorer):
    """P(y_i = 1) = 0.5 for all nodes."""

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return a zero logit for every node of every user, ``[num_users, num_nodes]``."""
        return torch.zeros(batch.num_users, batch.num_nodes, device=batch.visited.device)


class GlobalBaseline(FittedScorer):
    """P(y_i = 1) = global visit rate across all training graphs."""

    logit: torch.Tensor

    def __init__(self):
        super().__init__()

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Compute the global positive rate over the ``train`` users and store as a logit."""
        num_pos = train.visited.sum().item()
        num_total = train.visited.numel()

        p = torch.tensor(num_pos / num_total).clamp(1e-6, 1 - 1e-6)
        self.logit = inverse_sigmoid(p)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the global visit-rate logit for every node of every user, ``[num_users, num_nodes]``."""
        return self.logit.expand(batch.num_users, batch.num_nodes).to(batch.visited.device)


class VisitFrequencyBaseline(FittedScorer):
    """P(y_i = 1) = per-node visit frequency across the training users who do not live at the node."""

    logits: torch.Tensor

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Compute the home-excluded per-node visit rates over the ``train`` users and store them as logits."""
        self.logits = popularity_logit(train.visited, train.home_node)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the visit-frequency logit of every node for every user, ``[num_users, num_nodes]``."""
        return self.logits.to(batch.visited.device).expand(batch.num_users, -1)


class ConditionalVisitFrequencyBaseline(FittedScorer):
    """P(y_i = 1 | home = h) = per-home visit frequency, smoothed towards VisitFrequency.

    The smoothing covers nodes never visited by users with home = h.

    ``p(n | h) = (c_hn + a * p_n) / (u_h + a)``, with ``c_hn`` the training visits to ``n`` by residents of ``h``,
    ``u_h`` the training residents of ``h``, ``p_n`` the home-excluded visit rate, and ``a`` fit on validation.

    Attributes:
        strength: Fitted smoothing strength ``a``.
        logits: ``[num_nodes, num_nodes]`` smoothed logits indexed by (home, node).
    """

    strength: float
    logits: torch.Tensor

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Count visits per home over the ``train`` users and pick the smoothing strength on the ``val`` users."""
        prior = home_excluded_visit_rate(train.visited, train.home_node)
        counts, users = home_visit_counts(train.visited, train.home_node)
        val_prior = prior.expand_as(val.visited)

        self.strength = fit_smoothing_strength(counts, users, val, val_prior)
        self.logits = torch.logit(smoothed_rate(counts, users, prior, self.strength))

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return each user's home-conditioned visit-frequency logit of every node, ``[num_users, num_nodes]``."""
        return self.logits.to(batch.visited.device)[batch.home_node]
