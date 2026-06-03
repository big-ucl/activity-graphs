"""Unit tests for ActivityGraphModule."""

import types

import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule, extract_features
from activitygraphs.ml.models import NodeMLP


# Local reference implementations used only to cross-check the torchmetrics Retrieval* output.
# The project itself computes ranking metrics via torchmetrics, not these.
def precision_at_k(scores, labels, k):
    top_k = scores.topk(k).indices
    return labels[top_k].sum().item() / k


def recall_at_k(scores, labels, k):
    top_k = scores.topk(k).indices
    num_pos = labels.sum().int().item()
    return labels[top_k].sum().item() / num_pos if num_pos else 0.0


def ndcg_at_k(scores, labels, k):
    order = scores.argsort(descending=True)[:k]
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    dcg = (labels[order].float() * discounts).sum().item()
    ideal = torch.zeros(k)
    ideal[: min(int(labels.sum().item()), k)] = 1.0
    idcg = (ideal * discounts).sum().item()
    return dcg / idcg if idcg else 0.0


NUM_NODE_FEATURES = 6
NUM_DEMO_FEATURES = 3
NUM_EDGE_FEATURES = 2
IN_CHANNELS = NUM_NODE_FEATURES + NUM_DEMO_FEATURES


def make_batch(num_graphs: int = 2, num_nodes: int = 8, seed: int = 0, start_user_id: int = 0) -> pyg.data.Batch:
    """Return a synthetic PyG Batch with binary node labels, demographics, and unique user ids.

    Each graph carries ``graph_x`` so ``extract_features`` exercises the demographics path;
    feature width is ``NUM_NODE_FEATURES + NUM_DEMO_FEATURES`` (== IN_CHANNELS).
    """
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
        data = pyg.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            user_id=torch.tensor([start_user_id + g], dtype=torch.long),
        )
        data.graph_x = torch.rand(1, NUM_DEMO_FEATURES, generator=rng)
        graphs.append(data)
    return pyg.data.Batch.from_data_list(graphs)


def make_module(
    reg: str | None = None, lambda_reg: float = 0.01, pos_weight: float = 2.0, schedule_lr: bool = False
) -> ActivityGraphModule:
    model = NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1)
    return ActivityGraphModule(
        model=model, lr=1e-3, pos_weight=torch.tensor(pos_weight), reg=reg, lambda_reg=lambda_reg, schedule_lr=schedule_lr
    )


def force_positive_per_graph(batch: pyg.data.Batch) -> None:
    """Set the first node of each graph positive so ranking metrics are defined."""
    for i in range(batch.num_graphs):
        idx = (batch.batch == i).nonzero(as_tuple=True)[0][0]
        batch.y[idx] = 1.0


def capture_logs(module: ActivityGraphModule, monkeypatch) -> dict:
    """Patch ``self.log`` and ``self.log_dict`` (no Trainer attached) and collect everything logged."""
    logged: dict = {}
    monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.__setitem__(key, val))
    monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: logged.update(mapping))
    return logged


def attach_fake_trainer(module: ActivityGraphModule, overfit_batches: int = 0) -> ActivityGraphModule:
    """Give the module a minimal stand-in trainer so ``configure_optimizers`` can read ``overfit_batches``."""
    module._trainer = types.SimpleNamespace(overfit_batches=overfit_batches)
    return module


class TestTrainingStep:
    def test_returns_finite_scalar(self, monkeypatch):
        module = make_module()
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)
        loss = module.training_step(make_batch(), 0)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_l1_reg_increases_loss(self, monkeypatch):
        """L1 regularisation with large lambda_reg must yield strictly higher loss."""
        batch = make_batch(seed=1)

        base = make_module(reg=None)
        monkeypatch.setattr(base, "log", lambda *a, **kw: None)
        loss_base = base.training_step(batch, 0).item()

        module_l1 = make_module(reg="l1", lambda_reg=1.0)
        # copy identical weights so the only difference is the L1 term
        module_l1.model.load_state_dict(base.model.state_dict())
        monkeypatch.setattr(module_l1, "log", lambda *a, **kw: None)
        loss_l1 = module_l1.training_step(batch, 0).item()

        assert loss_l1 > loss_base

    def test_pos_weight_shifts_loss(self, monkeypatch):
        """Larger pos_weight must yield strictly higher loss on positive-heavy batches."""
        batch = make_batch(seed=2)
        batch.y[0] = 1.0  # ensure at least one positive label

        module_low = make_module(pos_weight=1.0)
        monkeypatch.setattr(module_low, "log", lambda *a, **kw: None)
        loss_low = module_low.training_step(batch, 0).item()

        # reuse identical weights, only the pos_weight differs
        module_high = ActivityGraphModule(model=module_low.model, lr=1e-3, pos_weight=torch.tensor(10.0))
        monkeypatch.setattr(module_high, "log", lambda *a, **kw: None)
        loss_high = module_high.training_step(batch, 0).item()

        assert loss_high > loss_low


class TestValidationStep:
    def test_val_metric_keys_logged(self, monkeypatch):
        """validation_step + on_validation_epoch_end must populate the BCE, ranking, and calibration keys."""
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)

        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        module.on_validation_epoch_end()

        k = module.k
        expected = {
            "val_bce",
            "val_bce_weighted",
            "val_r_precision",
            f"val_recall@{k}",
            f"val_ndcg@{k}",
            f"val_precision@{k}",
            "val_calibration_l1",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)

    def test_ranking_metrics_match_reference(self, monkeypatch):
        """torchmetrics precision/recall/ndcg must match the per-graph metrics.py functions.

        One graph == one user_id, so the Retrieval* grouping is per graph and the unweighted mean
        over non-empty groups equals the old per-graph average. MRR is excluded on purpose:
        RetrievalMRR uses first-relevant-rank, not mean-over-positives, so its value differs.
        """
        module = make_module()
        k = module.k
        batch = make_batch(num_graphs=4, num_nodes=8, seed=7)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            out = module(extract_features(batch, module.full_info), batch.edge_index, batch.edge_attr, batch.batch)

        precisions, recalls, ndcgs = [], [], []
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            scores = out[mask].squeeze()
            labels = batch.y[mask].squeeze()
            if labels.sum().int().item() == 0:
                continue
            precisions.append(float(precision_at_k(scores, labels, k)))
            recalls.append(float(recall_at_k(scores, labels, k)))
            ndcgs.append(float(ndcg_at_k(scores, labels, k)))

        ref = {
            f"val_precision@{k}": sum(precisions) / len(precisions),
            f"val_recall@{k}": sum(recalls) / len(recalls),
            f"val_ndcg@{k}": sum(ndcgs) / len(ndcgs),
        }

        with torch.no_grad():
            module.validation_step(batch, 0)
        actual = module.val_metrics.compute()

        for key, expected in ref.items():
            torch.testing.assert_close(actual[key].item(), expected, rtol=1e-5, atol=1e-6)

    def test_metrics_reset_between_epochs(self, monkeypatch):
        """on_validation_epoch_end must reset metric state, so a repeated epoch gives identical numbers."""
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8, seed=3)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        first = {key: val.item() for key, val in module.val_metrics.compute().items()}
        module.on_validation_epoch_end()  # logs + resets

        with torch.no_grad():
            module.validation_step(batch, 0)
        second = {key: val.item() for key, val in module.val_metrics.compute().items()}

        # Without a reset, re-feeding the same user_ids appends docs to existing groups and shifts
        # the result. Equality confirms state was cleared.
        assert first == second


class TestTestStep:
    def test_test_metric_keys_logged(self, monkeypatch):
        """test_step + on_test_epoch_end must populate the BCE, ranking, calibration, size, and sampled-set keys."""
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)

        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        k = module.k
        expected = {
            "test_bce",
            "test_bce_weighted",
            "test_r_precision",
            f"test_recall@{k}",
            f"test_ndcg@{k}",
            f"test_precision@{k}",
            "test_calibration_l1",
            "test_pred_size",
            "test_true_size",
            "test_sampled_recall",
            "test_sampled_size",
        }
        assert expected.issubset(logged.keys())


class TestConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self):
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_scheduler_monitors_val_bce(self):
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert result["lr_scheduler"]["monitor"] == "val_bce"

    def test_optimizer_is_adamw(self):
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_no_scheduler_when_not_requested(self):
        """With schedule_lr=False (the default), a bare optimizer is returned."""
        result = attach_fake_trainer(make_module()).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_no_scheduler_when_overfitting(self):
        """In overfit mode the scheduler is skipped and a bare optimizer is returned."""
        result = attach_fake_trainer(make_module(), overfit_batches=5).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)
