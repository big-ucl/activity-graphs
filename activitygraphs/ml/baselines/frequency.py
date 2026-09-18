"""Frequency-based (no learning) baselines for visit prediction."""

import torch
import torch_geometric as pyg

from activitygraphs.ml.baselines.common import (
    extract_per_user_tensors,
    fit_smoothing_strength,
    graph_home_indices,
    graph_node_indices,
    smoothed_rate,
)
from activitygraphs.ml.popularity import home_excluded_visit_rate, home_visit_counts, popularity_logit


def inverse_sigmoid(prob):
    """Return logit(prob) = log(prob / (1 - prob))."""
    return torch.log(prob / (1 - prob))


class UniformBaseline(torch.nn.Module):
    """P(y_i = 1) = 0.5 for all nodes."""

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return torch.zeros(x.shape[0], 1, device=x.device)


class GlobalBaseline(torch.nn.Module):
    """P(y_i = 1) = global visit rate across all training graphs."""

    def __init__(self):
        super().__init__()
        self.logit = None

    def fit(self, loader: pyg.loader.DataLoader):
        """Compute the global positive rate over ``loader`` and store as a logit."""
        num_pos = 0
        num_total = 0

        for batch in loader:
            num_pos += batch.y.sum().item()
            num_total += batch.y.numel()

        p = torch.tensor(num_pos / num_total).clamp(1e-6, 1 - 1e-6)
        self.logit = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.logit.expand(x.shape[0], 1).to(x.device)


class VisitFrequencyBaseline(torch.nn.Module):
    """P(y_i = 1) = per-node visit frequency across the training users who do not live at the node."""

    def __init__(self, num_nodes: int, is_home_idx: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_home_idx = is_home_idx
        self.logits = None

    def fit(self, loader: pyg.loader.DataLoader):
        """Compute the home-excluded per-node visit rates over ``loader`` and store them as logits."""
        per_user_tensors = extract_per_user_tensors(loader, self.is_home_idx)
        self.logits = popularity_logit(per_user_tensors.labels, per_user_tensors.home_idx)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.logits.to(x.device)[graph_node_indices(x, batch)].unsqueeze(1)


class ConditionalVisitFrequencyBaseline(torch.nn.Module):
    """P(y_i = 1 | home = h) = per-home visit frequency, smoothed to VisitFrequency is node is never visited by users
    with home = h.

    ``p(n | h) = (c_hn + a * p_n) / (u_h + a)``, with ``c_hn`` the training visits to ``n`` by residents of ``h``,
    ``u_h`` the training residents of ``h``, ``p_n`` the home-excluded visit rate, and ``a`` fit on validation.

    Attributes:
        strength: Fitted smoothing strength ``a``.
        logits: ``[num_nodes, num_nodes]`` smoothed logits indexed by (home, node).
    """

    def __init__(self, num_nodes: int, is_home_idx: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_home_idx = is_home_idx
        self.strength = None
        self.logits = None

    def fit(self, train_loader: pyg.loader.DataLoader, val_loader: pyg.loader.DataLoader):
        """Count visits per home on ``train_loader`` and pick the smoothing strength on ``val_loader``."""
        train = extract_per_user_tensors(train_loader, self.is_home_idx)
        val = extract_per_user_tensors(val_loader, self.is_home_idx)

        prior = home_excluded_visit_rate(train.labels, train.home_idx)
        counts, users = home_visit_counts(train.labels, train.home_idx)

        self.strength = fit_smoothing_strength(counts, users, val, prior.expand_as(val.labels))
        self.logits = torch.logit(smoothed_rate(counts, users, prior, self.strength))

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        homes = graph_home_indices(x, batch, self.is_home_idx)
        return self.logits.to(x.device)[homes, graph_node_indices(x, batch)].unsqueeze(1)
