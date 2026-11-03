"""Unit tests for the per-user row data path and the models scored from it, on the synthetic CMAP survey."""

import lightning as L
import polars as pl
import pytest
import torch
from test_input_snapshot import cache_survey_tensors

from activitygraphs.analysis import with_ranking_metrics
from activitygraphs.datasets import HouseholdSplit, Locations, SurveyDataModule, UserBatch
from activitygraphs.models.baselines import VisitFrequencyBaseline
from activitygraphs.models.baselines.popularity import popularity_logit
from activitygraphs.models.scorers import Scorer
from activitygraphs.tensors import tensors_path
from activitygraphs.training.lightning_module import RankingModule
from activitygraphs.training.losses import BCELoss

SPLIT = HouseholdSplit(val_size=0.2, test_size=0.2)
SPLIT_SEED = 42
BATCH_SIZE = 4
MAX_RECALL_K = 50
RECALL_KS = (1, 3, 5)


class FixedNodeScores(Scorer):
    """Scores every node of every user with the same fixed logits.

    Args:
        logits: Logit of each node, ``[num_nodes]``.
    """

    logits: torch.Tensor

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, locations: Locations, batch: UserBatch) -> torch.Tensor:
        """Return the logits of every node for every user of ``batch``, ``[num_users, num_nodes]``."""
        return self.logits.expand(batch.num_users, -1)


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


def per_user_avg_recall(module: RankingModule) -> pl.DataFrame:
    per_user = pl.DataFrame(module.per_user_columns)
    return with_ranking_metrics(per_user, MAX_RECALL_K).select("user_id", "avg_recall").sort("user_id")


def test_a_row_model_at_the_popularity_logits_reproduces_visit_frequency(cfg, tmp_path):
    datamodule = make_datamodule(cfg, tmp_path)
    train_users = datamodule.train_dataset.users
    locations = datamodule.locations

    baseline = VisitFrequencyBaseline(locations.num_nodes).fit(train_users)
    graph_module = RankingModule(
        scorer=baseline,
        locations=locations,
        lr=0.0,
        loss=BCELoss,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
    )

    popularity = popularity_logit(train_users.visited, train_users.home_node)
    grid_module = RankingModule(
        scorer=FixedNodeScores(popularity),
        locations=locations,
        lr=0.0,
        loss=BCELoss,
        max_recall_k=MAX_RECALL_K,
        recall_ks=RECALL_KS,
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
