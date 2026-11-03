"""Unit tests for the per-user grid data path and ``TwoTowerModule``, on the synthetic CMAP survey."""

import lightning as L
import polars as pl
import pytest
import torch
from test_input_snapshot import cache_survey_tensors

from activitygraphs.analysis import with_ranking_metrics
from activitygraphs.config import LossConfig
from activitygraphs.datasets import GraphBatch, HouseholdSplit, Locations, SurveyDataModule, UserBatch
from activitygraphs.ml.baselines import VisitFrequencyBaseline
from activitygraphs.ml.lightning_module import RankingModule
from activitygraphs.ml.losses import BCELoss, GridBCELoss, build_grid_loss, build_loss
from activitygraphs.ml.two_tower import NodeBias, TwoTowerModule
from activitygraphs.tensors import tensors_path

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
    cache_survey_tensors("cmap", tensors_path(cfg.data, tmp_path))
    return cfg


def make_datamodule(cfg, tmp_path) -> SurveyDataModule:
    datamodule = SurveyDataModule(cfg, SPLIT, SPLIT_SEED, BATCH_SIZE, num_workers=0, project_root=tmp_path)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def test_the_loader_yields_whole_rows_of_users(cfg, tmp_path):
    datamodule = make_datamodule(cfg, tmp_path)
    rows = next(iter(datamodule.test_dataloader()))
    batch = UserBatch.join(rows, datamodule.locations)

    assert batch.visited.shape == (batch.num_users, datamodule.locations.num_nodes)
    assert batch.num_users == min(BATCH_SIZE, len(datamodule.test_dataset))


def same_users_both_ways(cfg, tmp_path) -> tuple[GraphBatch, UserBatch, Locations]:
    datamodule = make_datamodule(cfg, tmp_path)
    train_dataset = datamodule.train_dataset
    locations = datamodule.locations
    rows = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))))
    batch = UserBatch.join(rows, locations)

    return GraphBatch.expand(batch, locations), batch, locations


@pytest.mark.parametrize("neg_sampler", ["uniform", "hard"])
def test_the_grid_bpr_loss_matches_the_graph_bpr_loss(cfg, tmp_path, neg_sampler):
    graph_batch, grid_batch, locations = same_users_both_ways(cfg, tmp_path)
    loss_cfg = bpr_config(neg_sampler)
    graph_loss_fn = build_loss(loss_cfg, locations.hop_distance, 0, MAX_RECALL_K).loss_fn
    grid_loss_fn = build_grid_loss(loss_cfg, MAX_RECALL_K).loss_fn

    scores = torch.randn(grid_batch.num_users, grid_batch.num_nodes, generator=torch.Generator().manual_seed(0))
    flat_scores = scores.reshape(-1, 1)

    graph_loss = graph_loss_fn(flat_scores, graph_batch, torch.Generator().manual_seed(1))
    grid_loss = grid_loss_fn(scores, grid_batch, torch.Generator().manual_seed(1))

    torch.testing.assert_close(grid_loss, graph_loss)


def per_user_avg_recall(module: RankingModule | TwoTowerModule) -> pl.DataFrame:
    per_user = pl.DataFrame(module.per_user_columns)
    return with_ranking_metrics(per_user, MAX_RECALL_K).select("user_id", "avg_recall").sort("user_id")


def test_the_bias_alone_reproduces_visit_frequency(cfg, tmp_path):
    datamodule = make_datamodule(cfg, tmp_path)
    train_users = datamodule.train_dataset.users
    locations = datamodule.locations
    is_home_idx = datamodule.schema.user_nodes.index_of("is_home")

    baseline = VisitFrequencyBaseline(locations.num_nodes).fit(train_users)
    graph_module = RankingModule(
        model=baseline,
        feature_encoder=None,
        locations=locations,
        lr=0.0,
        loss=BCELoss,
        is_home_idx=is_home_idx,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
    )

    bias = NodeBias.fit(train_users)
    grid_module = TwoTowerModule(
        model=bias,
        locations=locations,
        lr=0.0,
        loss=GridBCELoss,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
        weight_decay=0.0,
    )

    trainer = L.Trainer(logger=False, enable_progress_bar=False, enable_checkpointing=False, accelerator="cpu")
    (graph_results,) = trainer.test(graph_module, datamodule=datamodule, verbose=False)
    (grid_results,) = trainer.test(grid_module, datamodule=datamodule, verbose=False)

    graph_avg_recall = per_user_avg_recall(graph_module)
    grid_avg_recall = per_user_avg_recall(grid_module)

    assert grid_avg_recall.height > 0
    assert grid_avg_recall["user_id"].to_list() == graph_avg_recall["user_id"].to_list()
    assert (grid_avg_recall["avg_recall"] - graph_avg_recall["avg_recall"]).abs().max() < 5e-5
    assert grid_results == pytest.approx(graph_results)
