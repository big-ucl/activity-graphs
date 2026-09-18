"""Distance-decay baselines: nearest first, gravity with a Tanner deterrence, binned gravity and conditional gravity.

Every baseline here reads each node's distance from the user's home as the last column of ``x``, so they are evaluated
with ``full_info=True``. Distances are undone from the training ``StandardScaler`` back to metres.
"""

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler

from activitygraphs.ml.baselines.common import (
    extract_per_user_tensors,
    fit_smoothing_strength,
    graph_home_indices,
    graph_node_indices,
    non_home_mask,
    smoothed_rate,
)
from activitygraphs.ml.popularity import home_excluded_visit_rate, home_visit_counts

# A user's own home is at distance 0; clamping keeps its log-distance finite.
MIN_DISTANCE_M = 1.0

N_DISTANCE_BINS = 20
BIN_PSEUDOCOUNT = 0.5


def unscale_distances(scaled_distances: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    """Undo the distance ``StandardScaler``.

    Args:
        scaled_distances: Scaled distances, any shape.
        scaler: The distance scaler fitted on the training split.

    Returns:
        Distances in metres, clamped below at ``MIN_DISTANCE_M``.
    """
    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0])

    return (scaled_distances.float() * scale + mean).clamp(min=MIN_DISTANCE_M)


class DistanceDecayBaseline(torch.nn.Module):
    """Nearest first: scores each node by ``-log(d)``, with ``d`` its distance from home in metres.

    The scores are not probabilities.
    """

    def __init__(self, distance_scaler: StandardScaler):
        super().__init__()
        self.distance_scaler = distance_scaler

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return -unscale_distances(x[:, -1], self.distance_scaler).log().unsqueeze(1)


class GravityBaseline(torch.nn.Module):
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

    def __init__(self, is_home_idx: int, distance_scaler: StandardScaler, max_iter: int = 200):
        super().__init__()
        self.is_home_idx = is_home_idx
        self.distance_scaler = distance_scaler
        self.max_iter = max_iter
        self.popularity_logit = None
        self.d0 = None
        self.alpha = None
        self.beta_log = None
        self.beta_lin = None

    def fit(self, train_loader: pyg.loader.DataLoader):
        """Fit the intercept and the two decay parameters on ``train_loader`` with full-batch LBFGS."""
        split = extract_per_user_tensors(train_loader, self.is_home_idx)
        self.popularity_logit = torch.logit(home_excluded_visit_rate(split.labels, split.home_idx))

        pairs = non_home_mask(split)
        labels = split.labels[pairs].float()
        offset = self.popularity_logit.expand_as(split.labels)[pairs]
        distance = unscale_distances(split.distances, self.distance_scaler)[pairs]

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

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        distance = unscale_distances(x[:, -1], self.distance_scaler)
        return self.pair_logits(graph_node_indices(x, batch), distance).unsqueeze(1)


class GravityBinnedBaseline(torch.nn.Module):
    """Gravity model with a step-shaped deterrence fitted by counting: scores ``log p_n + log g_b(d)``.

    ``g_b = O_b / E_b``, with ``O_b`` the visited training pairs in distance bin ``b`` (plus a pseudocount) and ``E_b``
    the sum of ``p_n`` over the training pairs in the bin. The bin edges are quantiles of the visited pairs' distances.
    The scores are not probabilities, so its BCE is not meaningful.

    Attributes:
        log_popularity: ``log p_n``, ``[num_nodes]``.
        edges: Interior bin edges in metres, ``[n_bins - 1]``.
        log_deterrence: ``log g_b``, ``[n_bins]``.
    """

    def __init__(self, is_home_idx: int, distance_scaler: StandardScaler, n_bins: int = N_DISTANCE_BINS):
        super().__init__()
        self.is_home_idx = is_home_idx
        self.distance_scaler = distance_scaler
        self.n_bins = n_bins
        self.log_popularity = None
        self.edges = None
        self.log_deterrence = None

    def fit(self, train_loader: pyg.loader.DataLoader):
        """Count the observed and expected visits per distance bin on ``train_loader``."""
        split = extract_per_user_tensors(train_loader, self.is_home_idx)
        popularity = home_excluded_visit_rate(split.labels, split.home_idx)
        self.log_popularity = popularity.log()

        pairs = non_home_mask(split)
        visited = split.labels[pairs] > 0.5
        expected = popularity.expand_as(split.labels)[pairs]
        distance = unscale_distances(split.distances, self.distance_scaler)[pairs]

        quantiles = torch.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = torch.quantile(distance[visited], quantiles)
        bins = torch.bucketize(distance, self.edges)

        observed = torch.bincount(bins[visited], minlength=self.n_bins).float() + BIN_PSEUDOCOUNT
        expected = torch.zeros(self.n_bins).index_add_(0, bins, expected)
        self.log_deterrence = (observed / expected).log()

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        distance = unscale_distances(x[:, -1], self.distance_scaler)
        bins = torch.bucketize(distance, self.edges.to(x.device))
        popularity = self.log_popularity.to(x.device)[graph_node_indices(x, batch)]

        return (popularity + self.log_deterrence.to(x.device)[bins]).unsqueeze(1)


class ConditionalGravityBaseline(torch.nn.Module):
    """Per-home visit frequency smoothed toward a fitted gravity model.

    ``p(n | h) = (c_hn + a * g_hn) / (u_h + a)``, with ``g_hn = sigmoid(Gravity logit)`` for the pair and the smoothing
    strength ``a`` fitted on validation.

    Attributes:
        strength: Fitted smoothing strength ``a``.
        counts: Training visits indexed by (home, node), ``[num_nodes, num_nodes]``.
        users: Training residents per home, ``[num_nodes]``.
    """

    def __init__(self, gravity: GravityBaseline):
        super().__init__()
        self.gravity = gravity
        self.strength = None
        self.counts = None
        self.users = None

    def fit(self, train_loader: pyg.loader.DataLoader, val_loader: pyg.loader.DataLoader):
        """Count visits per home on ``train_loader`` and pick the smoothing strength on ``val_loader``."""
        is_home_idx = self.gravity.is_home_idx
        train = extract_per_user_tensors(train_loader, is_home_idx)
        val = extract_per_user_tensors(val_loader, is_home_idx)

        self.counts, self.users = home_visit_counts(train.labels, train.home_idx)

        node_idx = torch.arange(val.labels.shape[1]).expand_as(val.labels)
        val_distance = unscale_distances(val.distances, self.gravity.distance_scaler)
        val_prior = self.gravity.pair_logits(node_idx, val_distance).sigmoid()

        self.strength = fit_smoothing_strength(self.counts, self.users, val, val_prior)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        homes = graph_home_indices(x, batch, self.gravity.is_home_idx)
        nodes = graph_node_indices(x, batch)
        prior = self.gravity(x, edge_index, edge_attr, batch).sigmoid()

        counts = self.counts.to(x.device)[homes, nodes].unsqueeze(1)
        rate = smoothed_rate(counts, self.users.to(x.device)[homes], prior, self.strength)

        return torch.logit(rate)
