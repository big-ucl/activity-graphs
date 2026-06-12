import torch

from activitygraphs.ml.losses import bpr_loss

def test_bpr_zero_when_perfectly_ranked():
    # one user, positives all score far above negatives -> loss ~ 0
    logits = torch.tensor([10., 10., -10., -10.]).unsqueeze(-1)
    y = torch.tensor([1., 1., 0., 0.]).unsqueeze(-1)
    batch_index = torch.zeros(4, dtype=torch.long)
    assert bpr_loss(logits, y, batch_index, n_pairs=64).item() < 1e-3

def test_bpr_large_when_inverted():
    logits = torch.tensor([-10., -10., 10., 10.]).unsqueeze(-1)
    y = torch.tensor([1., 1., 0., 0.]).unsqueeze(-1)
    batch_index = torch.zeros(4, dtype=torch.long)
    assert bpr_loss(logits, y, batch_index, n_pairs=64).item() > 5.0

def test_bpr_invariant_to_per_user_constant():
    # the defining property (and why calibration is lost): adding c to a user's scores
    # leaves the loss unchanged
    logits = torch.randn(8, 1); y = (torch.rand(8, 1) > 0.5).float()
    batch_index = torch.zeros(8, dtype=torch.long); y[0] = 1; y[1] = 0
    g1 = torch.Generator().manual_seed(0); g2 = torch.Generator().manual_seed(0)
    a = bpr_loss(logits, y, batch_index, n_pairs=32, generator=g1)
    b = bpr_loss(logits + 3.7, y, batch_index, n_pairs=32, generator=g2)
    torch.testing.assert_close(a, b)

def test_bpr_skips_users_without_positives():
    # user 1 has no positives -> contributes nothing, no crash
    logits = torch.randn(8, 1)
    y = torch.tensor([1., 0., 1., 0., 0., 0., 0., 0.]).unsqueeze(-1)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert bpr_loss(logits, y, batch_index, n_pairs=16).isfinite()
