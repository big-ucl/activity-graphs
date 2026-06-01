"""Frequency-based (no learning) baselines for visit prediction."""

import torch
import torch_geometric as pyg


def inverse_sigmoid(prob):
    """Return logit(prob) = log(prob / (1 - prob))."""
    return torch.log(prob / (1 - prob))


def extract_is_home(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Return a boolean mask indicating home nodes (column ``is_home_idx > 0``)."""
    return (x[..., is_home_idx] > 0.0).bool()


def compute_ptr_from_batch(batch: torch.Tensor):
    ptr = torch.zeros(batch.max().item() + 2, dtype=torch.long, device=batch.device)
    ptr[1:] = batch.bincount().cumsum(0)

    return ptr


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


class NodeBaseline(torch.nn.Module):
    """P(y_i = 1) = per-node visit frequency across all training graphs."""

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.logits = None

    def fit(self, loader: pyg.loader.DataLoader):
        """Compute per-node visit rates over ``loader`` and store as logits."""
        visit_counts = torch.zeros(self.num_nodes)
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            node_indices = torch.arange(batch.num_nodes) - batch.ptr[batch.batch]
            visit_counts.scatter_add_(0, node_indices, batch.y.float().squeeze().cpu())
            graph_counts.scatter_add_(0, node_indices, torch.ones(batch.num_nodes))

        p = (visit_counts / graph_counts.clamp(min=1)).clamp(1e-6, 1 - 1e-6)
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        ptr = compute_ptr_from_batch(batch)
        node_indices = torch.arange(x.shape[0], device=x.device) - ptr[batch]
        return self.logits.to(x.device)[node_indices].unsqueeze(1)


class ConditionalNodeBaseline(torch.nn.Module):
    """P(y_i = 1 | home = h) = per-node visit frequency conditioned on home node."""

    def __init__(self, num_nodes: int, is_home_idx: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_home_idx = is_home_idx
        self.logits = None  # shape: [num_nodes, num_nodes] (node, home)

    def fit(self, loader: pyg.loader.DataLoader):
        """Compute per-node-per-home visit rates over ``loader`` and store as logits ``[num_nodes, num_nodes]``."""
        visit_counts = torch.zeros(self.num_nodes, self.num_nodes)
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            node_indices = torch.arange(batch.num_nodes) - batch.ptr[batch.batch]
            home_mask = extract_is_home(batch.x, self.is_home_idx)

            for i in range(batch.num_graphs):
                graph_mask = batch.batch == i
                home_nodes = home_mask[graph_mask].nonzero()

                if len(home_nodes) == 0:
                    continue

                home = home_nodes[0].item()
                visit_counts[node_indices[graph_mask], home] += batch.y[graph_mask].squeeze().float().cpu()
                graph_counts[home] += 1

        p = visit_counts / graph_counts.clamp(min=1).unsqueeze(0)
        p = p.clamp(1e-6, 1 - 1e-6)
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        ptr = compute_ptr_from_batch(batch)
        node_indices = torch.arange(x.shape[0], device=x.device) - ptr[batch]
        home_mask = extract_is_home(x, self.is_home_idx)
        home_indices = torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)

        for i in range(batch.max().item() + 1):
            graph_mask = batch == i
            home_nodes = home_mask[graph_mask].nonzero()

            home_indices[i] = 0 if len(home_nodes) == 0 else home_nodes[0].item()

        node_home = home_indices[batch]
        return self.logits.to(x.device)[node_indices, node_home.cpu()].unsqueeze(1).to(x.device)
