"""Distance-decay baselines: nearest first, gravity with a Tanner deterrence, binned gravity and conditional gravity.

Every baseline here reads each node's distance in metres from the user's home, ``batch.home_distance_m``.
"""

import torch
import torch.nn.functional as F

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.ml.baselines.common import Baseline, fit_smoothing_strength, non_home_mask, smoothed_rate
from activitygraphs.ml.popularity import home_excluded_visit_rate, home_visit_counts

# A user's own home is at distance 0; clamping keeps its log-distance finite.
MIN_DISTANCE_M = 1.0

N_DISTANCE_BINS = 20
BIN_PSEUDOCOUNT = 0.5


def clamp_distances(distances_m: torch.Tensor) -> torch.Tensor:
    """Return the distances in metres clamped below at ``MIN_DISTANCE_M``, shaped like ``distances_m``."""
    return distances_m.float().clamp(min=MIN_DISTANCE_M)


def extract_home_distances(batch: UserBatch) -> torch.Tensor:
    """Return the clamped distance in metres from home of every node for every user, ``[num_users, num_nodes]``."""
    return clamp_distances(batch.home_distance_m)


def expand_node_indices(batch: UserBatch) -> torch.Tensor:
    """Return the index of every node for every user, ``[num_users, num_nodes]``."""
    return torch.arange(batch.num_nodes, device=batch.visited.device).expand(batch.num_users, -1)


class DistanceDecayBaseline(Baseline):
    """Nearest first: scores each node by ``-log(d)``, with ``d`` its distance from home in metres.

    The scores are not probabilities.
    """

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return minus the log distance from home of every node for every user, ``[num_users, num_nodes]``."""
        return -extract_home_distances(batch).log()


class GravityBaseline(Baseline):
    """Gravity model with the home-excluded visit rate as mass and one Tanner deterrence for everyone.

    ``logit P(y = 1 | h, n) = alpha + logit(p_n) - beta_log * log(d / d0) - beta_lin * d / d0``, with ``d0`` the median
    distance of the visited training pairs. ``alpha``, ``beta_log`` and ``beta_lin`` are fitted by maximum likelihood on
    every training (user, non-home node) pair.

    Attributes:
        popularity_logit: ``logit(p_n)``, ``[num_nodes]``.
        d0: Distance scale in metres.
        alpha: Fitted intercept.
        beta_log: Fitted log-distance decay.
        beta_lin: Fitted linear-distance decay.
    """

    popularity_logit: torch.Tensor
    d0: float
    alpha: float
    beta_log: float
    beta_lin: float

    def __init__(self, max_iter: int = 200):
        super().__init__()
        self.max_iter = max_iter

    def fit(self, train: Users, distance_m: torch.Tensor):
        """Fit the intercept and the two decay parameters on the ``train`` users with full-batch LBFGS.

        Args:
            train: The training users.
            distance_m: Distance in metres between every pair of nodes, ``[num_nodes, num_nodes]``.
        """
        self.popularity_logit = torch.logit(home_excluded_visit_rate(train.visited, train.home_node))

        pairs = non_home_mask(train)
        labels = train.visited[pairs].float()
        offset = self.popularity_logit.expand_as(train.visited)[pairs]
        distance = clamp_distances(distance_m[train.home_node])[pairs]

        self.d0 = float(distance[labels > 0.5].median())
        relative = distance / self.d0
        log_relative = relative.log()

        params = torch.zeros(3, requires_grad=True)
        optimizer = torch.optim.LBFGS([params], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = params[0] + offset - params[1] * log_relative - params[2] * relative
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.alpha, self.beta_log, self.beta_lin = (float(value) for value in params.detach())

        return self

    def pair_logits(self, node_idx: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Compute the gravity logit of each (node, distance in metres) pair, shaped like ``node_idx``."""
        relative = distance / self.d0
        popularity = self.popularity_logit.to(distance.device)[node_idx]

        return self.alpha + popularity - self.beta_log * relative.log() - self.beta_lin * relative

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the gravity logit of every node for every user, ``[num_users, num_nodes]``."""
        return self.pair_logits(expand_node_indices(batch), extract_home_distances(batch))


class GravityBinnedBaseline(Baseline):
    """Gravity model with a step-shaped deterrence fitted by counting: scores ``log p_n + log g_b(d)``.

    ``g_b = O_b / E_b``, with ``O_b`` the visited training pairs in distance bin ``b`` (plus a pseudocount) and ``E_b``
    the sum of ``p_n`` over the training pairs in the bin. The bin edges are quantiles of the visited pairs' distances.
    The scores are not probabilities, so its BCE is not meaningful.

    Attributes:
        log_popularity: ``log p_n``, ``[num_nodes]``.
        edges: Interior bin edges in metres, ``[n_bins - 1]``.
        log_deterrence: ``log g_b``, ``[n_bins]``.
    """

    log_popularity: torch.Tensor
    edges: torch.Tensor
    log_deterrence: torch.Tensor

    def __init__(self, n_bins: int = N_DISTANCE_BINS):
        super().__init__()
        self.n_bins = n_bins

    def fit(self, train: Users, distance_m: torch.Tensor):
        """Count the observed and expected visits per distance bin over the ``train`` users.

        Args:
            train: The training users.
            distance_m: Distance in metres between every pair of nodes, ``[num_nodes, num_nodes]``.
        """
        popularity = home_excluded_visit_rate(train.visited, train.home_node)
        self.log_popularity = popularity.log()

        pairs = non_home_mask(train)
        visited = train.visited[pairs] > 0.5
        expected = popularity.expand_as(train.visited)[pairs]
        distance = clamp_distances(distance_m[train.home_node])[pairs]

        quantiles = torch.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = torch.quantile(distance[visited], quantiles)
        bins = torch.bucketize(distance, self.edges)

        observed = torch.bincount(bins[visited], minlength=self.n_bins).float() + BIN_PSEUDOCOUNT
        expected = torch.zeros(self.n_bins).index_add_(0, bins, expected)
        self.log_deterrence = (observed / expected).log()

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the binned-gravity logit of every node for every user, ``[num_users, num_nodes]``."""
        device = batch.visited.device
        bins = torch.bucketize(extract_home_distances(batch), self.edges.to(device))
        popularity = self.log_popularity.to(device)

        return popularity + self.log_deterrence.to(device)[bins]


class ConditionalGravityBaseline(Baseline):
    """Per-home visit frequency smoothed toward a fitted gravity model.

    ``p(n | h) = (c_hn + a * g_hn) / (u_h + a)``, with ``g_hn = sigmoid(Gravity logit)`` for the pair and the smoothing
    strength ``a`` fitted on validation.

    Attributes:
        strength: Fitted smoothing strength ``a``.
        counts: Training visits indexed by (home, node), ``[num_nodes, num_nodes]``.
        users: Training residents per home, ``[num_nodes]``.
    """

    strength: float
    counts: torch.Tensor
    users: torch.Tensor

    def __init__(self, gravity: GravityBaseline):
        super().__init__()
        self.gravity = gravity

    def fit(self, train: Users, val: Users, distance_m: torch.Tensor):
        """Count visits per home over the ``train`` users and pick the smoothing strength on the ``val`` users.

        Args:
            train: The training users.
            val: The validation users.
            distance_m: Distance in metres between every pair of nodes, ``[num_nodes, num_nodes]``.
        """
        self.counts, self.users = home_visit_counts(train.visited, train.home_node)

        node_idx = torch.arange(val.visited.shape[1]).expand_as(val.visited)
        val_distance = clamp_distances(distance_m[val.home_node])
        val_prior = self.gravity.pair_logits(node_idx, val_distance).sigmoid()

        self.strength = fit_smoothing_strength(self.counts, self.users, val, val_prior)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return each user's home-conditioned gravity logit of every node, ``[num_users, num_nodes]``."""
        device = batch.visited.device
        homes = batch.home_node
        prior = self.gravity(locations, batch).sigmoid()

        counts = self.counts.to(device)[homes]
        rate = smoothed_rate(counts, self.users.to(device)[homes], prior, self.strength)

        return torch.logit(rate)
