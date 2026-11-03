"""Unit tests for MetricsLogger and the aggregate frame built from its rows."""

import lightning as L
import polars as pl
import pytest
import torch
from encoders import make_encoder
from graphs import chain_edge_index, loader_of, make_locations, make_users

from activitygraphs.experiments.training import aggregate_frame
from activitygraphs.ml.lightning_module import RankingModule
from activitygraphs.ml.loggers import MetricsLogger
from activitygraphs.ml.losses import BCELoss
from activitygraphs.ml.models import NodeMLP

NUM_NODES = 6
NUM_FEATURES = 3


def make_survey(num_users: int = 8):
    """A chain graph, and users who live at node 0 and visit one other node each, with one numeric attribute."""
    generator = torch.Generator().manual_seed(0)
    edge_index = chain_edge_index(NUM_NODES)
    x = torch.rand(NUM_NODES, NUM_FEATURES - 1, generator=generator)
    edge_attr = torch.ones(edge_index.size(1), 1)
    random_walk_pe = torch.rand(NUM_NODES, 2, generator=generator)
    distance_m = torch.rand(NUM_NODES, NUM_NODES, generator=generator)
    locations = make_locations(x, edge_index, edge_attr, random_walk_pe, distance_m)

    visited = torch.zeros(num_users, NUM_NODES)
    for user in range(num_users):
        visited[user, [0, 1 + user % (NUM_NODES - 1)]] = 1.0
    home_node = torch.zeros(num_users, dtype=torch.long)
    users = make_users(visited, home_node, attributes=torch.zeros(num_users, 1))

    return locations, users


def fit_and_test(num_epochs: int) -> list[dict]:
    """Rows logged by a real ``Trainer`` fitting a small ``RankingModule`` for ``num_epochs`` and testing it."""
    locations, users = make_survey()
    encoder = make_encoder(NUM_FEATURES - 1, 1, 1)
    module = RankingModule(
        model=NodeMLP(num_layers=2, in_channels=encoder.out_dim, hidden_channels=4, out_channels=1),
        feature_encoder=encoder,
        locations=locations,
        lr=1e-2,
        loss=BCELoss,
        is_home_idx=0,
        max_recall_k=3,
        recall_ks=[1, 3],
    )
    loader = loader_of(users, batch_size=4)
    metrics_logger = MetricsLogger()
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="cpu",
        logger=[metrics_logger],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(module, dataloaders=loader, verbose=False)

    return metrics_logger.rows


class TestMetricsLogger:
    def test_merges_one_epochs_train_and_val_metrics_into_one_row(self):
        logger = MetricsLogger()

        logger.log_metrics({"val_bce": 0.4, "epoch": 3}, step=10)
        logger.log_metrics({"train_loss": 0.5, "epoch": 3}, step=10)

        assert logger.rows == [{"stage": "fit", "epoch": 3, "val_bce": 0.4, "train_loss": 0.5}]

    def test_test_metrics_form_one_row_with_a_null_epoch(self):
        logger = MetricsLogger()

        logger.log_metrics({"test_bce": 0.3, "test_r_precision": 0.5, "epoch": 7}, step=0)

        assert logger.rows == [{"stage": "test", "epoch": None, "test_bce": 0.3, "test_r_precision": 0.5}]

    def test_ignores_metrics_of_no_stage(self):
        logger = MetricsLogger()

        logger.log_metrics({"lr-AdamW": 0.01}, step=10)

        assert logger.rows == []

    def test_a_trainer_logs_every_epoch_and_the_test_after_the_epoch_end_hooks(self):
        """The ranking and calibration metrics are logged in or after the epoch-end hooks and must still arrive."""
        rows = fit_and_test(num_epochs=2)

        assert [(row["stage"], row["epoch"]) for row in rows] == [("fit", 0), ("fit", 1), ("test", None)]
        assert {"train_loss", "val_bce", "val_avg_recall@3", "val_calibration_l1"} <= set(rows[0])
        assert {"test_avg_recall@3", "test_recall@1", "test_calibration_l1", "test_n_scored_users"} <= set(rows[2])


class TestAggregateFrame:
    def _rows(self, num_epochs: int) -> list[dict]:
        logger = MetricsLogger()

        for epoch in range(num_epochs):
            logger.log_metrics({"train_loss": 0.5, "val_bce": 0.4, "epoch": epoch}, step=epoch)

        logger.log_metrics({"test_bce": 0.3, "test_r_precision": 0.5, "epoch": num_epochs}, step=0)

        return logger.rows

    @pytest.mark.parametrize("num_epochs", [50, 500])
    def test_test_metrics_survive_however_many_fit_rows_precede_them(self, num_epochs: int):
        frame = aggregate_frame(self._rows(num_epochs), "MLP")

        assert "test_r_precision" in frame.columns
        test_row = frame.filter(pl.col("stage") == "test")
        assert test_row["test_r_precision"].item() == pytest.approx(0.5)

    def test_every_row_carries_the_model_name(self):
        frame = aggregate_frame(self._rows(3), "MLP")

        assert frame["name"].to_list() == ["MLP"] * 4
