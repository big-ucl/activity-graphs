"""Loss and metric Protocol types, focal loss, and BCE helpers."""

import functools
from typing import Protocol

import torch
import torch.nn.functional as F


class LossFn(Protocol):
    """Protocol for loss functions that accept ``inputs``, ``targets``, and an optional ``pos_weight``."""

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, *, pos_weight: torch.Tensor | None = ..., **kwargs
    ) -> torch.Tensor: ...


class MetricFn(Protocol):
    """Protocol for metric functions that map ``(inputs, targets)`` to a scalar float."""

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> float: ...


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = -1,
) -> torch.Tensor:
    """Focal loss from https://arxiv.org/abs/1708.02002.

    Args:
        inputs: Raw logits of shape ``[N]`` or ``[N, 1]``.
        targets: Binary labels of the same shape.
        pos_weight: Optional positive-class BCE weight.
        gamma: Focusing parameter; higher values down-weight easy examples more.
        alpha: Class balance factor; set to ``-1`` to disable.
    """

    bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction="none")

    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    loss = focal_weight * bce

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
