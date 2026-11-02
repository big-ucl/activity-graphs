"""Unit tests for the per-user grid data path and ``TwoTowerModule``, on the synthetic CMAP survey."""

import lightning as L
import polars as pl
import pytest
import torch
import torch_geometric as pyg
from test_input_snapshot import build_dataset

from activitygraphs.analysis import with_ranking_metrics
from activitygraphs.config import LossConfig
from activitygraphs.datasets import HouseholdSplit, UserBatch, UserGridBatch, UserGridDataModule, UserGridDataset
from activitygraphs.datasets.datamodule import VisitDataModule
from activitygraphs.datasets.dataset import VisitDataset
from activitygraphs.ml.baselines import VisitFrequencyBaseline
from activitygraphs.ml.lightning_module import RankingModule
from activitygraphs.ml.losses import BCELoss, GridBCELoss, build_grid_loss, build_loss
from activitygraphs.ml.two_tower import NodeBias, TwoTowerModule

SPLIT = HouseholdSplit(val_size=0.2, test_size=0.2)
SPLIT_SEED = 42
BATCH_SIZE = 4
MAX_RECALL_K = 50
RECALL_KS = (1, 3, 5)


def bpr_config(neg_sampler: str) -> LossConfig:
    return LossConfig(type="bpr", n_pairs=16, neg_sampler=neg_sampler, is_near_threshold=1)


@pytest.fixture
def cfg(tmp_path, compose_config):
    cfg = compose_config("data=cmap")
    build_dataset("cmap", tmp_path / cfg.data.paths.tensors)
    return cfg


def make_datamodule(cfg, tmp_path) -> VisitDataModule:
    datamodule = VisitDataModule(cfg, SPLIT, SPLIT_SEED, BATCH_SIZE, num_workers=0, project_root=tmp_path)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def test_a_grid_item_holds_the_rows_of_the_graph_item(cfg, tmp_path):
    split = make_datamodule(cfg, tmp_path).test_dataset
    item = UserGridDataset(split)[0]
    graph_item = split[0]

    is_home_idx = split.schema.user_nodes.index_of("is_home")
    graph_home = graph_item.x_user_node[:, is_home_idx].argmax()

    assert item["user_id"] == graph_item.user_id
    assert item["home_node"] == graph_home
    torch.testing.assert_close(item["y"], graph_item.y.squeeze(-1))
    torch.testing.assert_close(item["distances"], graph_item.distances.squeeze(-1))
    torch.testing.assert_close(item["distances_m"], graph_item.distances_m.squeeze(-1))
    torch.testing.assert_close(item["user_attr"], graph_item.user_attr.squeeze(0))
    torch.testing.assert_close(item["user_attr_cat"], graph_item.user_attr_cat.squeeze(0))


def test_the_grid_loader_yields_whole_rows_of_users(cfg, tmp_path):
    datamodule = UserGridDataModule(make_datamodule(cfg, tmp_path))
    batch = UserGridBatch.from_dict(next(iter(datamodule.test_dataloader())))

    test_dataset = datamodule.visits.test_dataset

    assert batch.y.shape == (batch.num_users, test_dataset.num_nodes)
    assert batch.num_users == min(BATCH_SIZE, len(test_dataset))


def same_users_both_ways(cfg, tmp_path) -> tuple[UserBatch, UserGridBatch, VisitDataset]:
    split = make_datamodule(cfg, tmp_path).train_dataset
    num_users = len(split)

    graph_batch = UserBatch.from_pyg(next(iter(pyg.loader.DataLoader(split, batch_size=num_users))))
    grid_items = next(iter(torch.utils.data.DataLoader(UserGridDataset(split), batch_size=num_users)))

    return graph_batch, UserGridBatch.from_dict(grid_items), split


@pytest.mark.parametrize("neg_sampler", ["uniform", "hard"])
def test_the_grid_bpr_loss_matches_the_graph_bpr_loss(cfg, tmp_path, neg_sampler):
    graph_batch, grid_batch, split = same_users_both_ways(cfg, tmp_path)
    hop_distance = torch.as_tensor(split.locations.hop_distance)
    is_home_idx = split.schema.user_nodes.index_of("is_home")
    loss_cfg = bpr_config(neg_sampler)
    graph_loss_fn = build_loss(loss_cfg, hop_distance, is_home_idx, MAX_RECALL_K).loss_fn
    grid_loss_fn = build_grid_loss(loss_cfg, hop_distance, MAX_RECALL_K).loss_fn

    scores = torch.randn(grid_batch.num_users, grid_batch.num_nodes, generator=torch.Generator().manual_seed(0))
    flat_scores = scores.reshape(-1, 1)

    graph_loss = graph_loss_fn(flat_scores, graph_batch, torch.Generator().manual_seed(1))
    grid_loss = grid_loss_fn(scores, grid_batch, torch.Generator().manual_seed(1))

    torch.testing.assert_close(grid_loss, graph_loss)


def per_user_avg_recall(module: RankingModule | TwoTowerModule) -> pl.DataFrame:
    per_user = pl.DataFrame(module.per_user_columns)
    return with_ranking_metrics(per_user, MAX_RECALL_K).select("user_id", "avg_recall").sort("user_id")


def test_the_bias_alone_reproduces_visit_frequency(cfg, tmp_path):
    graph_datamodule = make_datamodule(cfg, tmp_path)
    grid_datamodule = UserGridDataModule(graph_datamodule)
    train_dataset = graph_datamodule.train_dataset
    home_hop_distance = train_dataset.locations.hop_distance
    is_home_idx = train_dataset.schema.user_nodes.index_of("is_home")

    baseline = VisitFrequencyBaseline(train_dataset.num_nodes, is_home_idx).fit(graph_datamodule.train_dataloader())
    graph_module = RankingModule(
        model=baseline,
        feature_encoder=None,
        lr=0.0,
        loss=BCELoss,
        is_home_idx=is_home_idx,
        home_hop_distance=home_hop_distance,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
    )

    bias = NodeBias.fit(train_dataset)
    grid_module = TwoTowerModule(
        model=bias,
        lr=0.0,
        loss=GridBCELoss,
        home_hop_distance=home_hop_distance,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
        weight_decay=0.0,
    )

    trainer = L.Trainer(logger=False, enable_progress_bar=False, enable_checkpointing=False, accelerator="cpu")
    (graph_results,) = trainer.test(graph_module, datamodule=graph_datamodule, verbose=False)
    (grid_results,) = trainer.test(grid_module, datamodule=grid_datamodule, verbose=False)

    graph_avg_recall = per_user_avg_recall(graph_module)
    grid_avg_recall = per_user_avg_recall(grid_module)

    assert grid_avg_recall.height > 0
    assert grid_avg_recall["user_id"].to_list() == graph_avg_recall["user_id"].to_list()
    assert (grid_avg_recall["avg_recall"] - graph_avg_recall["avg_recall"]).abs().max() < 5e-5
    assert grid_results == pytest.approx(graph_results)
