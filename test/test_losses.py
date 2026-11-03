import torch

from activitygraphs.config import LossConfig
from activitygraphs.ml.losses import bpr_loss, build_loss


def test_bpr_zero_when_perfectly_ranked():
    # one user, positives all score far above negatives -> loss ~ 0
    scores = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert bpr_loss(scores, y, n_pairs=64).item() < 1e-3


def test_bpr_large_when_inverted():
    scores = torch.tensor([[-10.0, -10.0, 10.0, 10.0]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert bpr_loss(scores, y, n_pairs=64).item() > 5.0


def test_bpr_invariant_to_per_user_constant():
    # the defining property (and why calibration is lost): adding c to a user's scores
    # leaves the loss unchanged
    scores = torch.randn(1, 8)
    y = (torch.rand(1, 8) > 0.5).float()
    y[0, 0] = 1
    y[0, 1] = 0
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = bpr_loss(scores, y, n_pairs=32, generator=g1)
    b = bpr_loss(scores + 3.7, y, n_pairs=32, generator=g2)
    torch.testing.assert_close(a, b)


def test_bpr_skips_users_without_positives():
    # user 1 has no positives -> contributes nothing, no crash
    scores = torch.randn(2, 4)
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert bpr_loss(scores, y, n_pairs=16).isfinite()


def test_hard_negatives_are_drawn_near_home_only():
    """With every far node scored above the positive, only the near negatives keep the hard loss small."""
    scores = torch.tensor([[5.0, 0.0, 10.0, 10.0]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    near_home = torch.tensor([[True, True, False, False]])

    assert bpr_loss(scores, y, n_pairs=64, neg_restrict=near_home).item() < 0.01
    assert bpr_loss(scores, y, n_pairs=64).item() > 1.0


def test_hard_negatives_fall_back_to_every_unvisited_node_when_none_is_near():
    scores = torch.tensor([[5.0, 0.0, 10.0, 10.0]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    nowhere = torch.zeros(1, 4, dtype=torch.bool)

    restricted = bpr_loss(scores, y, n_pairs=64, neg_restrict=nowhere, generator=torch.Generator().manual_seed(0))
    unrestricted = bpr_loss(scores, y, n_pairs=64, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(restricted, unrestricted)


def test_bpr_selects_checkpoints_on_the_average_recall_at_the_budget():
    loss = build_loss(LossConfig(type="bpr"), max_recall_k=50)
    assert (loss.monitor, loss.monitor_mode) == ("val_avg_recall@50", "max")


def test_bce_selects_checkpoints_on_the_validation_bce():
    loss = build_loss(LossConfig(type="bce"), max_recall_k=50)
    assert (loss.monitor, loss.monitor_mode) == ("val_bce", "min")
