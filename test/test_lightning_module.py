"""Unit tests for ActivityGraphModule."""

import pytest
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.models import NodeMLP

NUM_NODE_FEATURES = 6
NUM_EDGE_FEATURES = 2


def make_batch(num_graphs: int = 2, num_nodes: int = 5, seed: int = 0, start_user_id: int = 0) -> pyg.data.Batch:
    """Return a synthetic PyG Batch with binary node labels."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for g in range(num_graphs):
        x = torch.rand(num_nodes, NUM_NODE_FEATURES, generator=rng)
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()
        # simple chain edges
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr = torch.rand(edge_index.size(1), NUM_EDGE_FEATURES, generator=rng)
        user_id = torch.tensor([start_user_id + g], dtype=torch.long)
        graphs.append(pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, user_id=user_id))
    return pyg.data.Batch.from_data_list(graphs)


def make_module(reg: str | None = None, lambda_reg: float = 0.01) -> ActivityGraphModule:
    model = NodeMLP(
        num_layers=2,
        in_channels=NUM_NODE_FEATURES,
        hidden_channels=8,
        out_channels=1,
    )
    pos_weight = torch.tensor(2.0)
    return ActivityGraphModule(model=model, lr=1e-3, pos_weight=pos_weight, reg=reg, lambda_reg=lambda_reg)


class TestTrainingStep:
    def test_returns_finite_scalar(self, monkeypatch):
        module = make_module()
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)
        batch = make_batch()
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_l1_reg_increases_loss(self, monkeypatch):
        """L1 regularisation with large lambda_reg must yield strictly higher loss."""
        batch = make_batch(seed=1)

        module_base = make_module(reg=None)
        monkeypatch.setattr(module_base, "log", lambda *a, **kw: None)
        loss_base = module_base.training_step(batch, 0).item()

        module_l1 = make_module(reg="l1", lambda_reg=1.0)
        # copy identical weights so the only difference is the L1 term
        module_l1.model.load_state_dict(module_base.model.state_dict())
        monkeypatch.setattr(module_l1, "log", lambda *a, **kw: None)
        loss_l1 = module_l1.training_step(batch, 0).item()

        assert loss_l1 > loss_base

    def test_pos_weight_shifts_loss(self, monkeypatch):
        """Larger pos_weight must yield strictly higher loss on positive-heavy batches."""
        batch = make_batch(seed=2)
        # ensure at least one positive label
        batch.y[0] = 1.0

        module_low = ActivityGraphModule(
            model=NodeMLP(2, NUM_NODE_FEATURES, 8, 1),
            lr=1e-3,
            pos_weight=torch.tensor(1.0),
        )
        monkeypatch.setattr(module_low, "log", lambda *a, **kw: None)
        loss_low = module_low.training_step(batch, 0).item()

        module_high = ActivityGraphModule(
            model=module_low.model,
            lr=1e-3,
            pos_weight=torch.tensor(10.0),
        )
        monkeypatch.setattr(module_high, "log", lambda *a, **kw: None)
        loss_high = module_high.training_step(batch, 0).item()

        assert loss_high > loss_low


class TestValidationStep:
    def test_metrics_keys_logged(self, monkeypatch):
        """After validation_step + on_validation_epoch_end, all six expected keys must be logged."""
        logged = {}
        module = make_module()

        # ensure each graph has at least one positive label so ranking metrics are computed
        batch = make_batch(num_graphs=3, num_nodes=8)
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            batch.y[mask][0] = 1.0

        monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.update({key: val}))

        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        module.on_validation_epoch_end()

        k = module.k
        expected = {"val_bce", "val_bce_weighted", f"val_precision@{k}", f"val_recall@{k}", "val_mrr", f"val_ndcg@{k}"}
        assert expected.issubset(logged.keys())

    def test_val_outputs_cleared_after_epoch(self, monkeypatch):
        module = make_module()
        batch = make_batch()
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)
        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        module.on_validation_epoch_end()
        assert module._val_outputs == []


class TestConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self):
        module = make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_scheduler_monitors_val_bce(self):
        module = make_module()
        result = module.configure_optimizers()
        assert result["lr_scheduler"]["monitor"] == "val_bce"

    def test_optimizer_is_adamw(self):
        module = make_module()
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)
