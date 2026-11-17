"""Distance-decay baselines: nearest first and the gravity models.

The gravity models are gravity based on: distance (tanner, binned, conditional), travel-time, logsum of travel time.
Distance gravities read each node's distance in metres from the user's home, ``batch.home_distance_m``.
The travel-time and logsum gravity read the travel times from home, ``batch.home_travel_times_min``.
"""

import math
from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F

from activitygraphs.datasets.batch import UserBatch
from activitygraphs.datasets.locations import Locations
from activitygraphs.datasets.users import Users
from activitygraphs.models.baselines.common import fit_smoothing_strength, non_home_mask, smoothed_rate
from activitygraphs.models.baselines.popularity import home_excluded_visit_rate, home_visit_counts
from activitygraphs.models.scorers import FittedScorer
from activitygraphs.schema import ColumnKind, TensorSchema

# A user's own home is at distance 0; clamping keeps its log-distance finite.
MIN_DISTANCE_M = 1.0

# A pair of zones joined by a routed time of 0 minutes; clamping keeps its log-time finite.
MIN_TRAVEL_TIME_MIN = 1.0

LOGSUM_SCALE_GRID_PER_MIN = torch.logspace(-2, 0, 9)
"""Candidate logsum scales ``theta`` (unit: min^-1). The logsum gravity keeps the one with the best validation likelihood."""

N_DISTANCE_BINS = 20
BIN_PSEUDOCOUNT = 0.5


@dataclass(frozen=True)
class _HomeNodeCounts:
    """Counts of the training users per home and node, on which the gravity models are fitted.

    Row ``i`` is the home ``home_node_idx[i]``. ``num_visits[i, n]`` is the number of users of that home who visited
    node ``n``. ``num_residents[i, n]`` is the number of users living in that home, the same for every node ``n``. Both
    are 0 at the home's own node, so that node is left out of the fit, as it is in the evaluation.

    A gravity model gives every user of one home the same visit probability ``p`` for a node. The log-likelihood of
    those users for that node is then ``num_visits * log(p) + (num_residents - num_visits) * log(1 - p)``. A fit on
    these counts therefore gives the same result as a fit on every user and node, with far fewer rows.

    Attributes:
        home_node_idx: Node index of each home with at least one resident, ``[num_homes]``.
        num_visits: The number of users of home ``i`` who visited node ``n``, at ``[i, n]``. It is 0 at the home's
            own node, ``[i, home_node_idx[i]]``. ``[num_homes, num_nodes]``.
        num_residents: The number of users living in home ``i``, repeated along row ``i`` so that it weighs every
            ``[i, n]`` term of the loss. It is 0 at the home's own node, ``[i, home_node_idx[i]]``.
            ``[num_homes, num_nodes]``.
    """

    home_node_idx: torch.Tensor
    num_visits: torch.Tensor
    num_residents: torch.Tensor

    @classmethod
    def from_users(cls, users: Users) -> Self:
        """Return the counts of ``users`` per home and node."""
        num_visits_by_home, num_residents_by_home = home_visit_counts(users.visited, users.home_node)
        home_node_idx = torch.nonzero(num_residents_by_home > 0).squeeze(-1)
        is_home = F.one_hot(home_node_idx, num_visits_by_home.shape[1]).bool()
        num_visits = num_visits_by_home[home_node_idx].masked_fill(is_home, 0.0)
        num_residents = num_residents_by_home[home_node_idx].unsqueeze(-1).expand_as(is_home).masked_fill(is_home, 0.0)

        return cls(home_node_idx=home_node_idx, num_visits=num_visits, num_residents=num_residents)


class DistanceDecayBaseline(FittedScorer):
    """Nearest first: scores each node by ``-log(d)``, with ``d`` its distance from home in metres.

    The scores are not probabilities.
    """

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return minus the log distance from home of every node for every user, ``[num_users, num_nodes]``."""
        return -_extract_home_distances(batch).log()


class GravityBaseline(FittedScorer):
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

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Fit the intercept and the two decay parameters on the ``train`` users with full-batch LBFGS.

        The distances are read from ``locations.distance_m``.
        """
        self.popularity_logit = torch.logit(home_excluded_visit_rate(train.visited, train.home_node))
        counts = _HomeNodeCounts.from_users(train)

        distance = _clamp_distances(locations.distance_m[counts.home_node_idx]).unsqueeze(-1)
        self.d0 = float(_median_over_visits(distance, counts))

        features = _tanner_features(distance / self.d0)
        params = _fit_count_logit(features, self.popularity_logit, counts, self.max_iter)
        self.alpha, self.beta_log, self.beta_lin = (float(value) for value in params)

        return self

    def pair_logits(self, node_idx: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Compute the gravity logit of each (node, distance in metres) pair, shaped like ``node_idx``."""
        relative_distance = (distance / self.d0).unsqueeze(-1)
        popularity = self.popularity_logit.to(distance.device)[node_idx]
        beta_log = torch.tensor([self.beta_log])
        beta_lin = torch.tensor([self.beta_lin])

        return popularity + _tanner_logit(relative_distance, self.alpha, beta_log, beta_lin)

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the gravity logit of every node for every user, ``[num_users, num_nodes]``."""
        return self.pair_logits(_expand_node_indices(batch), _extract_home_distances(batch))


class TravelTimeGravityBaseline(FittedScorer):
    """Gravity model with the home-excluded visit rate as mass and a Tanner deterrence on the distance and on each travel time.

    ``logit P(y = 1 | h, n) = alpha + logit(p_n) - sum_k (beta_log_k * log(t_k / t0_k) + beta_lin_k * t_k / t0_k)
    + sum_m gamma_m * r_m``. The deterrence inputs ``t_k`` are the distance in metres and the time in minutes of each
    mode. ``t0_k`` is the median of ``t_k`` over the visited training pairs. ``r_m`` is the reachability flag of each
    mode. ``alpha``, ``beta_log``, ``beta_lin`` and ``gamma`` are fitted by maximum likelihood on every training (user,
    non-home node) pair.

    Args:
        travel_time_schema: The column schema of the travel times. The count columns of the schema are the times.
            The binary columns of the schema are the reachability flags.
        max_iter: Largest number of LBFGS iterations.

    Attributes:
        popularity_logit: ``logit(p_n)``, ``[num_nodes]``.
        scales: ``t0_k``, the median distance in metres, then the median time in minutes of each mode, over the visits
            of the training users, ``[num_deterrence_inputs]``.
        alpha: Fitted intercept.
        beta_log: Fitted log decay of the distance, then of the time of each mode, ``[num_deterrence_inputs]``.
        beta_lin: Fitted linear decay of the distance, then of the time of each mode, ``[num_deterrence_inputs]``.
        gamma: Fitted effect of each reachability flag, ``[num_reachability_flags]``.
    """

    popularity_logit: torch.Tensor
    scales: torch.Tensor
    alpha: float
    beta_log: torch.Tensor
    beta_lin: torch.Tensor
    gamma: torch.Tensor

    def __init__(self, travel_time_schema: TensorSchema, max_iter: int = 500):
        super().__init__()
        self.time_idx = travel_time_schema.indices(ColumnKind.COUNT)
        self.flag_idx = travel_time_schema.indices(ColumnKind.BINARY)
        self.max_iter = max_iter

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Fit the intercept, the decays and the flag effects on the ``train`` users with full-batch LBFGS.

        The distances are read from ``locations.distance_m`` and the travel times from ``locations.travel_times_min``.
        """
        visit_rates = home_excluded_visit_rate(train.visited, train.home_node)
        self.popularity_logit = torch.logit(visit_rates)
        counts = _HomeNodeCounts.from_users(train)

        home_distance_m = locations.distance_m[counts.home_node_idx]
        home_travel_times_min = locations.travel_times_min[counts.home_node_idx]
        deterrence, flags = self._split_inputs(home_distance_m, home_travel_times_min)
        self.scales = _median_over_visits(deterrence, counts)

        features = torch.cat([_tanner_features(deterrence / self.scales), flags], dim=-1)
        params = _fit_count_logit(features, self.popularity_logit, counts, self.max_iter)
        alpha, self.beta_log, self.beta_lin, self.gamma = _split_params(params, len(self.scales))
        self.alpha = float(alpha)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the travel-time gravity logit of every node for every user, ``[num_users, num_nodes]``."""
        device = batch.visited.device
        deterrence, flags = self._split_inputs(batch.home_distance_m, batch.home_travel_times_min)
        relative_inputs = deterrence / self.scales.to(device)
        popularity = self.popularity_logit.to(device)
        flag_effect = flags @ self.gamma.to(device)

        return popularity + _tanner_logit(relative_inputs, self.alpha, self.beta_log, self.beta_lin) + flag_effect

    def _split_inputs(
        self, distance_m: torch.Tensor, travel_times_min: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the distance and the time of each mode, then the reachability flags, from home to each node.

        Args:
            distance_m: Distance in metres of each pair, ``[...]``.
            travel_times_min: Travel times in minutes and reachability flags of each pair, following the travel-time
                schema, ``[..., num_travel_time_features]``.

        Returns:
            Tuple ``(deterrence, flags)``: the clamped distance then the clamped time of each mode,
            ``[..., num_deterrence_inputs]``, and the reachability flags, ``[..., num_reachability_flags]``.
        """
        distance = _clamp_distances(distance_m).unsqueeze(-1)
        times = travel_times_min[..., self.time_idx].float().clamp(min=MIN_TRAVEL_TIME_MIN)
        deterrence = torch.cat([distance, times], dim=-1)
        flags = travel_times_min[..., self.flag_idx].float()

        return deterrence, flags


class LogsumGravityBaseline(FittedScorer):
    """Gravity model with the home-excluded visit rate as mass and a Tanner deterrence on the logsum time of the modes.

    The logsum time of a pair is ``g = -log(mean_m exp(-theta * t_m)) / theta``, over the time ``t_m`` in minutes of
    each mode. It lies between the shortest and the mean time of the modes. The deterrence inputs ``x_k`` are the
    logsum time, then the distance in metres when ``with_distance`` is set.
    ``logit P(y = 1 | h, n) = alpha + logit(p_n) - sum_k (beta_log_k * log(x_k / x0_k) + beta_lin_k * x_k / x0_k)``,
    with ``x0_k`` the median of ``x_k`` over the visited training pairs. ``alpha``, ``beta_log`` and ``beta_lin`` are
    fitted by maximum likelihood on every training (user, non-home node) pair.

    ``theta`` is chosen first, on the logsum time alone: it is the value of ``LOGSUM_SCALE_GRID_PER_MIN`` with the best
    likelihood on the validation pairs. The deterrence, with the distance term when ``with_distance`` is set, is then
    fitted at that ``theta``. The distance term is left out of the choice, since the distance and the mean time of the
    modes are nearly collinear, and their decays then cancel.

    Args:
        travel_time_schema: The column schema of the travel times. The count columns of the schema are the times.
        with_distance: Add a Tanner term on the distance beside the logsum time.
        max_iter: Largest number of LBFGS iterations of one fit.

    Attributes:
        popularity_logit: ``logit(p_n)``, ``[num_nodes]``.
        theta: Chosen logsum scale, per minute.
        scales: ``x0_k``, the median logsum time in minutes, then the median distance in metres, over the visits of the
            training users, ``[num_deterrence_inputs]``.
        alpha: Fitted intercept.
        beta_log: Fitted log decay of the logsum time, then of the distance, ``[num_deterrence_inputs]``.
        beta_lin: Fitted linear decay of the logsum time, then of the distance, ``[num_deterrence_inputs]``.
    """

    popularity_logit: torch.Tensor
    theta: float
    scales: torch.Tensor
    alpha: float
    beta_log: torch.Tensor
    beta_lin: torch.Tensor

    def __init__(self, travel_time_schema: TensorSchema, with_distance: bool, max_iter: int = 500):
        super().__init__()
        self.time_idx = travel_time_schema.indices(ColumnKind.COUNT)
        self.with_distance = with_distance
        self.max_iter = max_iter

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Choose the logsum scale on the ``val`` users, then fit the intercept and the decays on the ``train`` users.

        The distances are read from ``locations.distance_m`` and the travel times from ``locations.travel_times_min``.
        """
        self.popularity_logit = torch.logit(home_excluded_visit_rate(train.visited, train.home_node))
        train_counts = _HomeNodeCounts.from_users(train)
        val_counts = _HomeNodeCounts.from_users(val)
        train_distance_m = locations.distance_m[train_counts.home_node_idx]
        train_travel_times_min = locations.travel_times_min[train_counts.home_node_idx]
        val_travel_times_min = locations.travel_times_min[val_counts.home_node_idx]

        self.theta = self._choose_theta(train_counts, val_counts, train_travel_times_min, val_travel_times_min)

        train_inputs = self._deterrence_inputs(train_distance_m, train_travel_times_min, self.theta)
        self.scales = _median_over_visits(train_inputs, train_counts)
        features = _tanner_features(train_inputs / self.scales)
        params = _fit_count_logit(features, self.popularity_logit, train_counts, self.max_iter)
        alpha, self.beta_log, self.beta_lin, _ = _split_params(params, len(self.scales))
        self.alpha = float(alpha)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the logsum gravity logit of every node for every user, ``[num_users, num_nodes]``."""
        deterrence = self._deterrence_inputs(batch.home_distance_m, batch.home_travel_times_min, self.theta)
        device = deterrence.device
        relative_inputs = deterrence / self.scales.to(device)
        popularity = self.popularity_logit.to(device)

        return popularity + _tanner_logit(relative_inputs, self.alpha, self.beta_log, self.beta_lin)

    def _choose_theta(
        self,
        train_counts: _HomeNodeCounts,
        val_counts: _HomeNodeCounts,
        train_travel_times_min: torch.Tensor,
        val_travel_times_min: torch.Tensor,
    ) -> float:
        """Return the logsum scale of ``LOGSUM_SCALE_GRID_PER_MIN`` with the best validation likelihood.

        Each candidate scale is scored by a Tanner fit on the logsum time alone.

        Args:
            train_counts: The counts of the training users per home and node.
            val_counts: The counts of the validation users per home and node.
            train_travel_times_min: Travel times from each training home to each node, following the travel-time
                schema, ``[num_train_homes, num_nodes, num_travel_time_features]``.
            val_travel_times_min: Travel times from each validation home to each node, following the travel-time
                schema, ``[num_val_homes, num_nodes, num_travel_time_features]``.

        Returns:
            The chosen logsum scale, per minute.
        """
        best_theta = math.nan
        best_val_loss = math.inf
        for theta in LOGSUM_SCALE_GRID_PER_MIN.tolist():
            train_time = self._logsum_times(train_travel_times_min, theta).unsqueeze(-1)
            scale = _median_over_visits(train_time, train_counts)
            features = _tanner_features(train_time / scale)
            params = _fit_count_logit(features, self.popularity_logit, train_counts, self.max_iter)

            val_time = self._logsum_times(val_travel_times_min, theta).unsqueeze(-1)
            val_logits = self.popularity_logit + _tanner_features(val_time / scale) @ params
            val_loss = float(_count_loss(val_logits, val_counts))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_theta = theta

        return best_theta

    def _deterrence_inputs(
        self, distance_m: torch.Tensor, travel_times_min: torch.Tensor, theta: float
    ) -> torch.Tensor:
        """Return the logsum time from home to each node, then the distance when ``with_distance`` is set.

        Args:
            distance_m: Distance in metres of each pair, ``[...]``.
            travel_times_min: Travel times in minutes and reachability flags of each pair, following the travel-time
                schema, ``[..., num_travel_time_features]``.
            theta: Logsum scale, per minute.

        Returns:
            The logsum time in minutes, then the distance in metres clamped below at ``MIN_DISTANCE_M``,
            ``[..., num_deterrence_inputs]``.
        """
        logsum_time = self._logsum_times(travel_times_min, theta).unsqueeze(-1)
        if not self.with_distance:
            return logsum_time

        distance = _clamp_distances(distance_m).unsqueeze(-1)

        return torch.cat([logsum_time, distance], dim=-1)

    def _logsum_times(self, travel_times_min: torch.Tensor, theta: float) -> torch.Tensor:
        """Return the logsum time in minutes from home to each node, clamped below at ``MIN_TRAVEL_TIME_MIN``.

        Args:
            travel_times_min: Travel times in minutes and reachability flags of each pair, following the travel-time
                schema, ``[..., num_travel_time_features]``.
            theta: Logsum scale, per minute.

        Returns:
            Logsum times, ``[...]``.
        """
        mode_times = travel_times_min[..., self.time_idx].float()
        log_mean = torch.logsumexp(-theta * mode_times, dim=-1) - math.log(len(self.time_idx))

        return (-log_mean / theta).clamp(min=MIN_TRAVEL_TIME_MIN)


class GravityBinnedBaseline(FittedScorer):
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

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Count the observed and expected visits per distance bin over the ``train`` users.

        The distances are read from ``locations.distance_m``.
        """
        popularity = home_excluded_visit_rate(train.visited, train.home_node)
        self.log_popularity = popularity.log()

        pairs = non_home_mask(train)
        visited = train.visited[pairs] > 0.5
        expected = popularity.expand_as(train.visited)[pairs]
        distance = _clamp_distances(locations.distance_m[train.home_node])[pairs]

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
        bins = torch.bucketize(_extract_home_distances(batch), self.edges.to(device))
        popularity = self.log_popularity.to(device)

        return popularity + self.log_deterrence.to(device)[bins]


class ConditionalGravityBaseline(FittedScorer):
    """Per-home visit frequency smoothed toward a fitted gravity model.

    ``p(n | h) = (c_hn + a * g_hn) / (u_h + a)``, with ``g_hn = sigmoid(Gravity logit)`` for the pair and the smoothing
    strength ``a`` fitted on validation.

    Attributes:
        strength: Fitted smoothing strength ``a``.
        num_visits: ``c_hn``, the training visits indexed by (home, node), ``[num_nodes, num_nodes]``.
        num_residents: ``u_h``, the training residents of each home, ``[num_nodes]``.
    """

    strength: float
    num_visits: torch.Tensor
    num_residents: torch.Tensor

    def __init__(self, gravity: GravityBaseline):
        super().__init__()
        self.gravity = gravity

    def fit(self, train: Users, val: Users, locations: Locations) -> Self:
        """Count visits per home over the ``train`` users and pick the smoothing strength on the ``val`` users.

        The distances are read from ``locations.distance_m``.
        """
        self.num_visits, self.num_residents = home_visit_counts(train.visited, train.home_node)

        node_idx = torch.arange(val.visited.shape[1]).expand_as(val.visited)
        val_distance = _clamp_distances(locations.distance_m[val.home_node])
        val_prior = self.gravity.pair_logits(node_idx, val_distance).sigmoid()

        self.strength = fit_smoothing_strength(self.num_visits, self.num_residents, val, val_prior)

        return self

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return each user's home-conditioned gravity logit of every node, ``[num_users, num_nodes]``."""
        device = batch.visited.device
        home_node_idx = batch.home_node
        prior = self.gravity(locations, batch).sigmoid()

        num_visits = self.num_visits.to(device)[home_node_idx]
        num_residents = self.num_residents.to(device)[home_node_idx]
        rate = smoothed_rate(num_visits, num_residents, prior, self.strength)

        return torch.logit(rate)


def _clamp_distances(distance_m: torch.Tensor) -> torch.Tensor:
    """Return the distances in metres clamped below at ``MIN_DISTANCE_M``, shaped like ``distance_m``."""
    return distance_m.float().clamp(min=MIN_DISTANCE_M)


def _extract_home_distances(batch: UserBatch) -> torch.Tensor:
    """Return the clamped distance in metres from home of every node for every user, ``[num_users, num_nodes]``."""
    return _clamp_distances(batch.home_distance_m)


def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return the lower weighted median of each column of ``values``, ``[num_columns]``.

    Args:
        values: Values of each row, ``[num_rows, num_columns]``.
        weights: Non-negative weight of each row, ``[num_rows]``. A row of weight ``w`` counts as ``w`` rows.
    """
    sorted_values, order = values.sort(dim=0)
    cumulative_weights = weights[order].cumsum(dim=0)
    half_weight = torch.ceil(weights.sum() / 2)
    median_idx = torch.searchsorted(
        cumulative_weights.T.contiguous(), half_weight.expand(values.shape[1], 1).contiguous()
    )

    return sorted_values.gather(0, median_idx.T).squeeze(0)


def _tanner_features(relative_inputs: torch.Tensor) -> torch.Tensor:
    """Return the features of the Tanner logit from home to each node: 1, then ``-log(r)``, then ``-r`` for each input.

    ``r`` is an input divided by its median, such as ``d / d0`` for the distance. The features come in the order of
    the coefficients read by ``_split_params``: the intercept, the log decay of each input, then the linear decay of
    each input.

    Args:
        relative_inputs: Each input divided by its median over the training visits, ``[..., num_inputs]``.

    Returns:
        Features, ``[..., 1 + 2 * num_inputs]``.
    """
    intercept = torch.ones_like(relative_inputs[..., :1])

    return torch.cat([intercept, -relative_inputs.log(), -relative_inputs], dim=-1)


def _tanner_logit(
    relative_inputs: torch.Tensor, alpha: float, beta_log: torch.Tensor, beta_lin: torch.Tensor
) -> torch.Tensor:
    """Return the Tanner logit from home to each node, from the fitted intercept and decays.

    Args:
        relative_inputs: Each input divided by its median over the training visits, ``[..., num_inputs]``.
        alpha: Fitted intercept.
        beta_log: Fitted log decay of each input, ``[num_inputs]``.
        beta_lin: Fitted linear decay of each input, ``[num_inputs]``.

    Returns:
        Logits, without the popularity offset, ``[...]``.
    """
    intercept = torch.tensor([alpha])
    coefficients = torch.cat([intercept, beta_log, beta_lin]).to(relative_inputs)

    return _tanner_features(relative_inputs) @ coefficients


def _median_over_visits(inputs: torch.Tensor, counts: _HomeNodeCounts) -> torch.Tensor:
    """Return the median of each input over the visits counted in ``counts``, ``[num_inputs]``.

    Args:
        inputs: Inputs of each home and node, ``[num_homes, num_nodes, num_inputs]``.
        counts: The counts per home and node.
    """
    return _weighted_median(inputs.flatten(0, 1), counts.num_visits.flatten())


def _count_loss(logits: torch.Tensor, counts: _HomeNodeCounts) -> torch.Tensor:
    """Return the mean Bernoulli loss per user and node, computed from the counts per home and node, ``[]``.

    Args:
        logits: Logit of each home and node, ``[num_homes, num_nodes]``.
        counts: The counts per home and node.
    """
    visit_rates = counts.num_visits / counts.num_residents.clamp(min=1.0)
    loss = F.binary_cross_entropy_with_logits(logits, visit_rates, weight=counts.num_residents, reduction="sum")

    return loss / counts.num_residents.sum()


def _fit_count_logit(
    features: torch.Tensor, offset: torch.Tensor, counts: _HomeNodeCounts, max_iter: int
) -> torch.Tensor:
    """Return the coefficients of the logit ``offset + features @ coefficients``, fitted on ``counts`` with LBFGS.

    Args:
        features: Features of each home and node, ``[num_homes, num_nodes, num_coefficients]``.
        offset: Popularity logit of each node, not fitted, ``[num_nodes]``.
        counts: The counts per home and node.
        max_iter: Largest number of LBFGS iterations.

    Returns:
        Fitted coefficients, ``[num_coefficients]``.
    """
    coefficients = torch.zeros(features.shape[-1], requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficients], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = _count_loss(offset + features @ coefficients, counts)
        loss.backward()
        return loss

    optimizer.step(closure)

    return coefficients.detach()


def _split_params(
    params: torch.Tensor, num_inputs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the intercept, the log decays, the linear decays and the flag effects held in ``params``, in that order."""
    alpha = params[0]
    beta_log = params[1 : 1 + num_inputs]
    beta_lin = params[1 + num_inputs : 1 + 2 * num_inputs]
    gamma = params[1 + 2 * num_inputs :]

    return alpha, beta_log, beta_lin, gamma


def _expand_node_indices(batch: UserBatch) -> torch.Tensor:
    """Return the index of every node for every user, ``[num_users, num_nodes]``."""
    return torch.arange(batch.num_nodes, device=batch.visited.device).expand(batch.num_users, -1)
