"""Stochastic location-sampling strategies (Poisson, PPS) over model logits."""

import torch
import torch_geometric as pyg


def unweight_probs(probs: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Invert the pos_weight tilt: recover calibrated p from weighted-BCE-trained sigmoid output."""
    w = pos_weight
    return probs / (w - (w - 1) * probs)


def poisson_sampling(
    logits: torch.Tensor, generator: torch.Generator | None, pos_weight: float | None = None
) -> torch.Tensor:
    """Sample each node independently via a Bernoulli draw on sigmoid(logits); gradient is detached."""
    probs = torch.sigmoid(logits)

    if pos_weight is not None:
        probs = unweight_probs(probs, pos_weight)

    return torch.bernoulli(probs, generator=generator).detach()


def pps_sampling(
    num_samples: int, logits: torch.Tensor, batch: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Probability-proportional-to-size sampling: draw exactly ``num_samples`` nodes per graph in the batch."""
    logits, mask = pyg.utils.to_dense_batch(logits, batch, fill_value=0.0)
    probs = torch.sigmoid(logits).squeeze(-1) * mask

    weights = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    indices = torch.multinomial(weights, num_samples, replacement=False, generator=generator)

    samples = torch.zeros_like(probs)
    samples.scatter_(dim=1, index=indices, value=1.0)

    return samples[mask].unsqueeze(-1)
