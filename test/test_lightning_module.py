"""Unit tests for RankingModule."""

import types

import lightning as L
import pytest
import torch
import torch_geometric as pyg
from encoders import make_encoder
from torchmetrics import Metric

from activitygraphs.ml.features import HomeAnchoredPE, RandomWalkPE
from activitygraphs.ml.lightning_module import RankingModule
from activitygraphs.ml.losses import BCELoss
from activitygraphs.ml.models import NodeMLP

MAX_RECALL_K = 50
LR_SCHEDULER = {"factor": 0.5, "patience": 5}
RECALL_KS = [1, 3, 5]


def recall_at_k(scores, labels, k):
    """Reference recall@k of one tie-free user, from an explicit top-k."""
    top_k = scores.topk(min(k, scores.numel())).indices
    return labels[top_k].sum().item() / labels.sum().item()


NUM_LOCATION_FEATURES = 5
NUM_USER_NODE_FEATURES = 1
NUM_NODE_FEATURES = NUM_LOCATION_FEATURES + NUM_USER_NODE_FEATURES
NUM_DEMO_FEATURES = 3
NUM_EDGE_FEATURES = 2
NUM_RWPE_FEATURES = 4
IN_CHANNELS = NUM_NODE_FEATURES + NUM_DEMO_FEATURES
HOME_PE_BINS = 8
HOME_PE_MAX_HOPS = 40
# Column of the per-user ``is_home`` indicator in the synthetic ``x_user_node``, mirroring the real dataset.
IS_HOME_IDX = 0
HOME_NODE = 0


def chain_hop_distance(num_nodes: int = 8) -> torch.Tensor:
    """Hop distances of the chain graphs ``make_batch`` builds: nodes i and j are ``|i - j|`` hops apart."""
    idx = torch.arange(num_nodes)
    return (idx[:, None] - idx[None, :]).abs().float()


def make_batch(num_graphs: int = 2, num_nodes: int = 8, seed: int = 0, start_user_id: int = 0) -> pyg.data.Batch:
    """Return a synthetic PyG Batch with binary node labels, user attributes, and unique user ids.

    Each graph carries ``user_attr`` so the encoder exercises the user attribute path;
    feature width is ``NUM_NODE_FEATURES + NUM_DEMO_FEATURES`` (== IN_CHANNELS). Node ``HOME_NODE``
    of every graph is that user's home and is labelled visited, as it always is in the real data.
    """
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for g in range(num_graphs):
        x_loc = torch.rand(num_nodes, NUM_LOCATION_FEATURES, generator=rng)
        x_user_node = torch.zeros(num_nodes, NUM_USER_NODE_FEATURES)
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()
        x_user_node[HOME_NODE, IS_HOME_IDX] = 1.0
        y[HOME_NODE] = 1.0
        # simple chain edges
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr = torch.rand(edge_index.size(1), NUM_EDGE_FEATURES, generator=rng)
        data = pyg.data.Data(
            x_loc=x_loc,
            x_user_node=x_user_node,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            user_id=torch.tensor([start_user_id + g], dtype=torch.long),
        )
        data.user_attr = torch.rand(1, NUM_DEMO_FEATURES, generator=rng)
        data.random_walk_pe = torch.rand(num_nodes, NUM_RWPE_FEATURES, generator=rng)
        graphs.append(data)
    return pyg.data.Batch.from_data_list(graphs)


def make_default_encoder():
    """The encoder of the location and user-node features and the demographics of ``make_batch``."""
    return make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, NUM_DEMO_FEATURES)


def make_module(
    lr_scheduler: dict | None = None,
    store_score_vectors: bool = False,
) -> RankingModule:
    model = NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1)
    return RankingModule(
        model=model,
        feature_encoder=make_default_encoder(),
        lr=1e-3,
        loss=BCELoss,
        lr_scheduler=lr_scheduler,
        store_score_vectors=store_score_vectors,
        is_home_idx=IS_HOME_IDX,
        home_hop_distance=chain_hop_distance(),
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
    )


def force_positive_per_graph(batch: pyg.data.Batch) -> None:
    """Set a non-home node of each graph positive so ranking metrics survive the home exclusion."""
    for i in range(batch.num_graphs):
        idx = (batch.batch == i).nonzero(as_tuple=True)[0][HOME_NODE + 1]
        batch.y[idx] = 1.0


def scored_nodes(batch: pyg.data.Batch, graph: int) -> torch.Tensor:
    """Row indices of the nodes of one graph that the ranking metrics score."""
    rows = (batch.batch == graph).nonzero(as_tuple=True)[0]
    return rows[rows != rows[HOME_NODE]]


def metric_value(value):
    """Return ``value``, computed if it is a ``Metric``, as Lightning logs it."""
    return value.compute() if isinstance(value, Metric) else value


def capture_logs(module: RankingModule, monkeypatch) -> dict:
    """Patch ``self.log`` and ``self.log_dict`` (no Trainer attached) and collect everything logged.

    A logged ``Metric`` is computed when it is logged, which equals Lightning's epoch-end value once every batch has
    been updated.
    """
    logged: dict = {}
    monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.__setitem__(key, metric_value(val)))
    monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: logged.update(mapping))
    return logged


def attach_fake_trainer(module: RankingModule, overfit_batches: int = 0) -> RankingModule:
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

        expected = {
            "val_bce",
            "val_r_precision",
            f"val_avg_recall@{MAX_RECALL_K}",
            *(f"val_recall@{k}" for k in RECALL_KS),
            "val_calibration_l1",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)

    def test_ranking_metrics_match_reference(self, monkeypatch):
        """The logged average recall and recall@k must match an explicit top-k over each user's scored nodes.

        The reference is computed on the scored nodes only, since the home node is dropped from both the candidates
        and the labels.
        """
        module = make_module()
        batch = make_batch(num_graphs=4, num_nodes=8, seed=7)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            out = module.compute_logits(batch)

        per_user = []
        for i in range(batch.num_graphs):
            mask = scored_nodes(batch, i)
            scores = out[mask].squeeze().sigmoid()
            labels = batch.y[mask].squeeze()
            recall_curve = [recall_at_k(scores, labels, k) for k in range(1, MAX_RECALL_K + 1)]
            per_user.append({
                f"val_avg_recall@{MAX_RECALL_K}": sum(recall_curve) / MAX_RECALL_K,
                **{f"val_recall@{k}": recall_curve[k - 1] for k in RECALL_KS},
            })

        ref = {key: sum(user[key] for user in per_user) / len(per_user) for key in per_user[0]}

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
        """test_step + on_test_epoch_end must populate exactly the BCE, ranking, user-count and calibration keys."""
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)

        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        expected = {
            "test_bce",
            "test_r_precision",
            f"test_avg_recall@{MAX_RECALL_K}",
            *(f"test_recall@{k}" for k in RECALL_KS),
            "test_n_scored_users",
            "test_n_dropped_users",
            "test_calibration_l1",
        }
        assert set(logged.keys()) == expected


class TestUseDemographics:
    """The model is sized from the encoder, so the demographics are applied at run time exactly when sized for."""

    def _module(self, use_demographics: bool) -> RankingModule:
        num_demographics = NUM_DEMO_FEATURES if use_demographics else 0
        encoder = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, num_demographics)
        model = NodeMLP(num_layers=2, in_channels=encoder.out_dim, hidden_channels=8, out_channels=1)

        return RankingModule(
            model=model,
            feature_encoder=encoder,
            lr=1e-3,
            loss=BCELoss,
            is_home_idx=IS_HOME_IDX,
            home_hop_distance=chain_hop_distance(),
            max_recall_k=MAX_RECALL_K,
            recall_ks=RECALL_KS,
        )

    def test_demographics_are_concatenated_by_default(self):
        batch = make_batch(num_graphs=2, num_nodes=8)

        assert self._module(use_demographics=True).compute_logits(batch).shape == (16, 1)

    def test_a_model_built_without_demographics_runs_without_them(self):
        """Without the runtime flag this raises a shape mismatch, which is what makes the ablation runnable."""
        batch = make_batch(num_graphs=2, num_nodes=8)

        assert self._module(use_demographics=False).compute_logits(batch).shape == (16, 1)

    def test_the_two_arms_see_different_features(self):
        batch = make_batch(num_graphs=2, num_nodes=8)

        with_demo = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, NUM_DEMO_FEATURES)(batch)
        without_demo = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, 0)(batch)

        assert with_demo.shape[1] == without_demo.shape[1] + NUM_DEMO_FEATURES


class TestRwpe:
    """The random-walk PE is appended to the model input when ``use_rwpe`` is set, and only then."""

    def _module(self) -> RankingModule:
        rwpe = RandomWalkPE(NUM_RWPE_FEATURES)
        encoder = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, NUM_DEMO_FEATURES, rwpe)
        model = NodeMLP(num_layers=2, in_channels=encoder.out_dim, hidden_channels=8, out_channels=1)

        return RankingModule(
            model=model,
            feature_encoder=encoder,
            lr=1e-3,
            loss=BCELoss,
            is_home_idx=IS_HOME_IDX,
            home_hop_distance=chain_hop_distance(),
            max_recall_k=MAX_RECALL_K,
            recall_ks=RECALL_KS,
        )

    def test_appends_the_encoding_after_x(self):
        batch = make_batch(num_graphs=2, num_nodes=8)
        rwpe = RandomWalkPE(NUM_RWPE_FEATURES)
        x = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, 0, rwpe)(batch)

        torch.testing.assert_close(x, torch.cat([batch.x_loc, batch.x_user_node, batch.random_walk_pe], dim=-1))

    def test_left_out_without_its_source(self):
        batch = make_batch(num_graphs=2, num_nodes=8)
        x = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, 0)(batch)

        torch.testing.assert_close(x, torch.cat([batch.x_loc, batch.x_user_node], dim=-1))

    def test_a_model_built_with_rwpe_trains(self, monkeypatch):
        batch = make_batch(num_graphs=2, num_nodes=8)
        module = self._module()
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)

        assert torch.isfinite(module.training_step(batch, 0))


class TestTrainRanking:
    def _module(self, **kwargs) -> RankingModule:
        return RankingModule(
            model=NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1),
            feature_encoder=make_default_encoder(),
            lr=1e-3,
            loss=BCELoss,
            is_home_idx=IS_HOME_IDX,
            home_hop_distance=chain_hop_distance(),
            max_recall_k=MAX_RECALL_K,
            recall_ks=RECALL_KS,
            **kwargs,
        )

    def test_logs_average_recall_on_the_training_batch_by_default(self, monkeypatch):
        """Paired with `val_avg_recall@K` this is the train/val ranking gap, which the BPR loss cannot give."""
        module = self._module()
        batch = make_batch(num_graphs=2, num_nodes=8)
        force_positive_per_graph(batch)
        logged = capture_logs(module, monkeypatch)

        module.training_step(batch, 0)

        assert f"train_avg_recall@{MAX_RECALL_K}" in logged
        assert "train_loss" in logged

    def test_can_be_switched_off(self, monkeypatch):
        """The per-user loop costs ~8% of epoch time, so TTS-scale splits need to be able to skip it."""
        module = self._module(log_train_ranking=False)
        batch = make_batch(num_graphs=2, num_nodes=8)
        force_positive_per_graph(batch)
        logged = capture_logs(module, monkeypatch)

        module.training_step(batch, 0)

        assert module.train_avg_recall is None
        assert f"train_avg_recall@{MAX_RECALL_K}" not in logged
        assert "train_loss" in logged


class TestScoreVectors:
    def _tested(self, monkeypatch, store_score_vectors: bool):
        module = make_module(store_score_vectors=store_score_vectors)
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        return module, batch

    def test_not_collected_by_default(self, monkeypatch):
        module, _ = self._tested(monkeypatch, store_score_vectors=False)
        assert module.per_user_score_vectors == []

    def test_one_vector_per_node_per_scored_user(self, monkeypatch):
        module, _ = self._tested(monkeypatch, store_score_vectors=True)

        vectors = module.per_user_score_vectors
        assert len(vectors) == len(module.per_user_columns["user_id"])
        assert all(len(vector) == 8 for vector in vectors)

    def test_vectors_reproduce_the_reported_r_precision(self, monkeypatch):
        """The stored scores must be the quantity actually ranked, in node order, or the health checks lie.

        The vector covers every node, the home one included, so the reported score is recovered by
        dropping the home node from it first - the same exclusion the module applies.
        """
        module, batch = self._tested(monkeypatch, store_score_vectors=True)

        for row, user_id in enumerate(module.per_user_columns["user_id"]):
            graph = (batch.user_id == user_id).nonzero(as_tuple=True)[0].item()
            keep = torch.ones(batch.num_nodes // batch.num_graphs, dtype=torch.bool)
            keep[HOME_NODE] = False

            target = batch.y.squeeze(-1)[batch.batch == graph][keep]
            scores = torch.tensor(module.per_user_score_vectors[row])[keep]
            r = int(target.sum())

            top_r = scores.topk(r).indices
            assert float(target[top_r].sum() / r) == pytest.approx(module.per_user_columns["r_precision"][row])


class TestPositiveHops:
    """Each retained positive carries its hop distance from the user's home node."""

    def _tested(self, monkeypatch):
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8, seed=7)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        return module, batch

    def test_hops_follow_the_chain_distance_from_home(self, monkeypatch):
        """On the chain graphs node ``i`` is ``i`` hops from the home node 0."""
        module, batch = self._tested(monkeypatch)
        columns = module.per_user_columns

        for row, user_id in enumerate(columns["user_id"]):
            graph = (batch.user_id == user_id).nonzero(as_tuple=True)[0].item()
            positives = batch.y.squeeze(-1)[batch.batch == graph].nonzero(as_tuple=True)[0].tolist()
            expected = [float(node) for node in positives if node != HOME_NODE]

            assert columns["pos_hops"][row] == expected

    def test_flags_reproduce_the_reported_scores(self, monkeypatch):
        module, _ = self._tested(monkeypatch)
        columns = module.per_user_columns

        for row, n_pos in enumerate(columns["n_pos"]):
            assert sum(columns["pos_in_top_r"][row]) / n_pos == pytest.approx(columns["r_precision"][row])


class TestHomeExclusion:
    """The home node is neither a candidate nor a member of ``RG_i`` when scoring."""

    def _tested(self, monkeypatch):
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8, seed=11)
        force_positive_per_graph(batch)
        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        return module, batch, logged

    def test_the_home_node_is_dropped_from_the_realised_set(self, monkeypatch):
        module, batch, _ = self._tested(monkeypatch)

        for row, user_id in enumerate(module.per_user_columns["user_id"]):
            graph = (batch.user_id == user_id).nonzero(as_tuple=True)[0].item()
            full = int(batch.y.squeeze(-1)[batch.batch == graph].sum())

            assert module.per_user_columns["n_pos_home_incl"][row] == full
            assert module.per_user_columns["n_pos"][row] == full - 1

    def test_a_model_that_only_finds_home_scores_zero(self, monkeypatch):
        """The naive headline is ~90% home identification, so ranking home first must now buy nothing."""
        module = make_module()
        batch = make_batch(num_graphs=2, num_nodes=8, seed=5)
        capture_logs(module, monkeypatch)

        # Each user visits their home and exactly one other node.
        is_home = batch.x_user_node[:, IS_HOME_IDX] > 0.0
        batch.y = is_home.clone().float().unsqueeze(-1)
        for i in range(batch.num_graphs):
            batch.y[scored_nodes(batch, i)[0]] = 1.0

        # Home on top, and the one visited node ranked below every unvisited one.
        home_only = -batch.y.clone()
        home_only[is_home] = 10.0
        monkeypatch.setattr(module, "compute_logits", lambda _batch: home_only)

        with torch.no_grad():
            module.test_step(batch, 0)

        assert module.test_metrics["test_r_precision"].compute().item() == pytest.approx(0.0)

    def test_logs_how_many_users_the_headline_covers_and_drops(self, monkeypatch):
        """One user realises only their own home, so the exclusion leaves them nothing to rank."""
        module = make_module()
        batch = make_batch(num_graphs=2, num_nodes=8, seed=3)
        force_positive_per_graph(batch)

        # Strip user 1 back to a home-only realised set: their home stays positive, nothing else.
        batch.y[scored_nodes(batch, 1)] = 0.0

        logged = capture_logs(module, monkeypatch)
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        assert logged["test_n_scored_users"] == 1
        assert logged["test_n_dropped_users"] == 1

    def test_nothing_is_dropped_when_every_user_keeps_a_non_home_node(self, monkeypatch):
        _, _, logged = self._tested(monkeypatch)

        assert logged["test_n_scored_users"] == 3
        assert logged["test_n_dropped_users"] == 0

    def test_the_counts_are_logged_on_validation_too(self, monkeypatch):
        module = make_module()
        batch = make_batch(num_graphs=2, num_nodes=8, seed=7)
        force_positive_per_graph(batch)
        logged = capture_logs(module, monkeypatch)

        with torch.no_grad():
            module.validation_step(batch, 0)
        module.on_validation_epoch_end()

        assert logged["val_n_scored_users"] == 2
        assert logged["val_n_dropped_users"] == 0


class TestRankingView:
    """A ranking metric added later has to inherit the exclusion from the view, not re-apply a mask."""

    def _step(self, monkeypatch):
        module = make_module()
        batch = make_batch(num_graphs=2, num_nodes=8, seed=13)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        with torch.no_grad():
            return module._common_evaluation_step(batch, "test"), batch

    def test_the_ranking_view_covers_the_scored_candidates_only(self, monkeypatch):
        step, batch = self._step(monkeypatch)
        n_scored = batch.num_nodes - batch.num_graphs

        for values in (step.ranking.logits, step.ranking.probs, step.ranking.target, step.ranking.users):
            assert values.shape == (n_scored,)

    def test_the_full_tensors_still_cover_every_node(self, monkeypatch):
        """The loss, the BCE diagnostics and the sanity metric read these on purpose."""
        step, batch = self._step(monkeypatch)

        for values in (step.probs, step.target, step.users):
            assert values.shape == (batch.num_nodes,)

    def test_no_home_node_survives_into_the_ranking_view(self, monkeypatch):
        step, batch = self._step(monkeypatch)
        home_rows = (batch.x_user_node[:, IS_HOME_IDX] > 0.0).nonzero(as_tuple=True)[0]

        assert not step.ranking.scored[home_rows].any()
        # Every user still has candidates left, so none of them drops out of the view.
        assert set(step.ranking.users.tolist()) == set(batch.user_id.tolist())


class TestConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self):
        result = attach_fake_trainer(make_module(lr_scheduler=LR_SCHEDULER)).configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_scheduler_monitors_val_bce(self):
        result = attach_fake_trainer(make_module(lr_scheduler=LR_SCHEDULER)).configure_optimizers()
        assert result["lr_scheduler"]["monitor"] == "val_bce"

    def test_optimizer_is_adamw(self):
        result = attach_fake_trainer(make_module(lr_scheduler=LR_SCHEDULER)).configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_no_scheduler_when_not_requested(self):
        """With no lr_scheduler (the default), a bare optimizer is returned."""
        result = attach_fake_trainer(make_module()).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_no_scheduler_when_overfitting(self):
        """In overfit mode the scheduler is skipped and a bare optimizer is returned."""
        result = attach_fake_trainer(make_module(lr_scheduler=LR_SCHEDULER), overfit_batches=5).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_scheduler_takes_the_given_settings(self):
        result = attach_fake_trainer(make_module(lr_scheduler={"factor": 0.25, "patience": 3})).configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]

        assert (scheduler.factor, scheduler.patience) == (0.25, 3)


class TestCalibration:
    def test_both_stages_use_the_given_bins(self):
        module = RankingModule(
            model=NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1),
            feature_encoder=make_default_encoder(),
            lr=1e-3,
            loss=BCELoss,
            is_home_idx=IS_HOME_IDX,
            home_hop_distance=chain_hop_distance(),
            max_recall_k=MAX_RECALL_K,
            recall_ks=RECALL_KS,
            calibration_bins=7,
        )

        assert module.val_calibration.n_bins == 7
        assert module.test_calibration.n_bins == 7


class TestCheckpoint:
    """A checkpoint must rebuild the module from its own hyperparameters plus the objects it was not given."""

    def test_hyperparameters_leave_out_the_objects_passed_in(self):
        hparams = make_module(lr_scheduler=LR_SCHEDULER).hparams

        assert hparams["lr"] == 1e-3
        assert hparams["lr_scheduler"] == LR_SCHEDULER
        assert not {"model", "encoder", "loss", "home_hop_distance"} & set(hparams)

    def test_compiling_keeps_the_state_dict_keys(self):
        module = make_module()
        module.compile_model = True
        module._trainer = types.SimpleNamespace(fast_dev_run=False, overfit_batches=0)
        keys = list(module.state_dict())

        module.configure_model()

        assert list(module.state_dict()) == keys

    def test_reloads_from_a_checkpoint(self, tmp_path):
        module = make_module(lr_scheduler=LR_SCHEDULER)
        path = tmp_path / "best.ckpt"
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": L.__version__,
            },
            path,
        )

        model = NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1)
        reloaded = RankingModule.load_from_checkpoint(
            path, model=model, encoder=make_default_encoder(), loss=BCELoss, home_hop_distance=chain_hop_distance()
        )

        assert reloaded.lr_scheduler == LR_SCHEDULER
        for key, value in module.state_dict().items():
            assert torch.equal(reloaded.state_dict()[key], value)


def make_home_batch(
    num_graphs: int = 2, num_nodes: int = 8, seed: int = 0, home_nodes: list[int] | None = None
) -> tuple[pyg.data.Batch, torch.Tensor, int]:
    """A batch with a clean one-hot ``is_home`` column plus a chain-graph hop-distance matrix.

    Column 0 of ``x_user_node`` is ``is_home`` (one node per graph set to 1). Because the graphs
    are chains 0-1-...-(N-1), hop distance between nodes i and j is ``|i - j|``, so ``home_hop_distance``
    is the ``[N, N]`` absolute-difference matrix. ``home_nodes`` picks each graph's home node index.
    """
    batch = make_batch(num_graphs=num_graphs, num_nodes=num_nodes, seed=seed)
    is_home_idx = 0
    batch.x_user_node[:, is_home_idx] = 0.0
    if home_nodes is None:
        home_nodes = [0] * num_graphs
    for i, h in enumerate(home_nodes):
        graph_rows = (batch.batch == i).nonzero(as_tuple=True)[0]
        batch.x_user_node[graph_rows[h], is_home_idx] = 1.0

    return batch, chain_hop_distance(num_nodes), is_home_idx


def make_home_pe_module(home_hop_distance: torch.Tensor, is_home_idx: int, n_bins: int = HOME_PE_BINS) -> RankingModule:
    """A module with the home-anchored PE enabled; model width accounts for the extra ``n_bins`` columns."""
    home_pe = HomeAnchoredPE(home_hop_distance, is_home_idx, n_bins, HOME_PE_MAX_HOPS)
    encoder = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, NUM_DEMO_FEATURES, home_pe)
    model = NodeMLP(num_layers=2, in_channels=encoder.out_dim, hidden_channels=8, out_channels=1)
    return RankingModule(
        model=model,
        feature_encoder=encoder,
        lr=1e-3,
        loss=BCELoss,
        home_hop_distance=home_hop_distance,
        is_home_idx=is_home_idx,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
    )


class TestHomePositionalEncoding:
    def test_shape_and_home_is_peak(self):
        """Encoding is [num_nodes, n_bins]; the home node (hop 0) peaks in the first RBF bin at value 1."""
        batch, hhd, is_home_idx = make_home_batch()
        home_pe = HomeAnchoredPE(hhd, is_home_idx, 8, HOME_PE_MAX_HOPS)
        pe = home_pe(batch)

        assert pe.shape == (batch.num_nodes, 8)
        home_rows = batch.x_user_node[:, is_home_idx] > 0.0
        assert torch.all(pe[home_rows].argmax(dim=1) == 0)
        torch.testing.assert_close(pe[home_rows].amax(dim=1), torch.ones(int(home_rows.sum())))

    def test_matches_rbf_formula(self):
        """Explicit RBF over hop-to-home reproduces the encoding exactly (single graph, home = node 0)."""
        batch, hhd, is_home_idx = make_home_batch(num_graphs=1, num_nodes=8, home_nodes=[0])
        n_bins = 8
        home_pe = HomeAnchoredPE(hhd, is_home_idx, n_bins, HOME_PE_MAX_HOPS)
        pe = home_pe(batch)

        hops = hhd[0].clamp(max=40)  # row 0 == distances from the home node
        centers = torch.linspace(0, 40, n_bins)
        expected = torch.exp(-((hops.unsqueeze(-1) - centers) ** 2) / 2.0)
        torch.testing.assert_close(pe, expected)

    def test_hops_are_capped_at_max_hops(self):
        """Every node at or beyond ``max_hops`` gets the encoding of ``max_hops`` itself."""
        batch, hhd, is_home_idx = make_home_batch(num_graphs=1, num_nodes=8, home_nodes=[0])
        home_pe = HomeAnchoredPE(hhd, is_home_idx, 4, 3)
        pe = home_pe(batch)

        torch.testing.assert_close(pe[3:], pe[3].expand(5, 4))
        assert not torch.allclose(pe[2], pe[3])

    def test_varies_per_user(self):
        """Different home anchors give different encodings for the same node positions (the whole point)."""
        batch, hhd, is_home_idx = make_home_batch(home_nodes=[0, 7])
        home_pe = HomeAnchoredPE(hhd, is_home_idx, 8, HOME_PE_MAX_HOPS)
        pe = home_pe(batch)
        assert not torch.allclose(pe[batch.batch == 0], pe[batch.batch == 1])

    def test_the_encoder_appends_bins_columns(self):
        """The encoder widens by exactly ``bins`` when the home PE is a source, and ``out_dim`` follows."""
        batch, hhd, is_home_idx = make_home_batch()
        home_pe = HomeAnchoredPE(hhd, is_home_idx, HOME_PE_BINS, HOME_PE_MAX_HOPS)
        base = make_default_encoder()
        with_pe = make_encoder(NUM_LOCATION_FEATURES, NUM_USER_NODE_FEATURES, NUM_DEMO_FEATURES, home_pe)

        assert with_pe(batch).shape[1] - base(batch).shape[1] == HOME_PE_BINS
        assert with_pe.out_dim - base.out_dim == HOME_PE_BINS

    def test_training_step_is_finite(self, monkeypatch):
        """A full training_step with the PE enabled returns a finite scalar loss."""
        batch, hhd, is_home_idx = make_home_batch()
        module = make_home_pe_module(hhd, is_home_idx, n_bins=8)
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)

        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.isfinite()
