"""Unit tests for ml callbacks (EpochMetricsCollector, HopBandTableLogger)."""

import types
from unittest.mock import MagicMock

import polars as pl
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from activitygraphs.ml.callbacks import EpochMetricsCollector, HopBandTableLogger
from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.models import NodeMLP
from activitygraphs.ml.training import aggregate_frame


def fake_trainer(callback_metrics: dict, sanity_checking: bool = False, current_epoch: int = 0):
    """Minimal stand-in trainer exposing the attributes the collector reads."""
    return types.SimpleNamespace(
        callback_metrics={key: torch.tensor(val) for key, val in callback_metrics.items()},
        sanity_checking=sanity_checking,
        current_epoch=current_epoch,
    )


class TestEpochMetricsCollector:
    def test_fit_row_keeps_train_and_val_only(self):
        collector = EpochMetricsCollector()
        trainer = fake_trainer(
            {"train_loss": 0.5, "val_bce": 0.4, "val_precision@5": 0.2, "test_bce": 0.9}, current_epoch=3
        )

        collector.on_train_epoch_end(trainer, None)

        (row,) = collector.rows
        assert row["stage"] == "fit"
        assert row["epoch"] == 3
        assert row["train_loss"] == pytest.approx(0.5)
        assert row["val_bce"] == pytest.approx(0.4)
        assert "test_bce" not in row

    def test_test_row_keeps_test_only_with_null_epoch(self):
        collector = EpochMetricsCollector()
        # Ranking metrics (test_r_precision etc.) are only in callback_metrics by on_test_end, after the
        # module's on_test_epoch_end runs; the collector must read them there, not at epoch end.
        trainer = fake_trainer({"val_bce": 0.4, "test_bce": 0.3, "test_precision@5": 0.25, "test_r_precision": 0.5})

        collector.on_test_end(trainer, None)

        (row,) = collector.rows
        assert row["stage"] == "test"
        assert row["epoch"] is None
        assert row["test_bce"] == pytest.approx(0.3)
        assert row["test_r_precision"] == pytest.approx(0.5)
        assert "val_bce" not in row

    def test_sanity_check_epoch_skipped(self):
        collector = EpochMetricsCollector()
        trainer = fake_trainer({"val_bce": 0.4}, sanity_checking=True)

        collector.on_train_epoch_end(trainer, None)

        assert collector.rows == []

    def test_fit_row_skipped_when_no_train_or_val_metrics(self):
        collector = EpochMetricsCollector()
        trainer = fake_trainer({"lr": 0.01})  # no train_/val_ keys

        collector.on_train_epoch_end(trainer, None)

        assert collector.rows == []


class TestAggregateFrame:
    def _collect(self, num_epochs: int) -> list[dict]:
        collector = EpochMetricsCollector()

        for epoch in range(num_epochs):
            collector.on_train_epoch_end(fake_trainer({"train_loss": 0.5, "val_bce": 0.4}, current_epoch=epoch), None)

        collector.on_test_end(fake_trainer({"test_bce": 0.3, "test_r_precision": 0.5}), None)

        return collector.rows

    @pytest.mark.parametrize("num_epochs", [50, 500])
    def test_test_metrics_survive_however_many_fit_rows_precede_them(self, num_epochs: int):
        frame = aggregate_frame(self._collect(num_epochs), "MLP")

        assert "test_r_precision" in frame.columns
        test_row = frame.filter(pl.col("stage") == "test")
        assert test_row["test_r_precision"].item() == pytest.approx(0.5)

    def test_every_row_carries_the_model_name(self):
        frame = aggregate_frame(self._collect(3), "MLP")

        assert frame["name"].to_list() == ["MLP"] * 4


class TestHopBandTableLogger:
    def _make_module(self, rows: list[dict]) -> ActivityGraphModule:
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(model=model, lr=1e-3, pos_weight=torch.tensor(1.0))
        module.hop_band_rows = rows
        return module

    def test_logs_table_of_hop_band_rows(self):
        rows = [
            {"hop_band": "0-2", "hop_low": 0, "k": 5, "ndcg": 0.5, "n_pos": 3.0},
            {"hop_band": "3-5", "hop_low": 3, "k": 5, "ndcg": 0.2, "n_pos": 7.0},
        ]
        trainer = types.SimpleNamespace(logger=MagicMock(spec=WandbLogger))

        HopBandTableLogger().on_test_end(trainer, self._make_module(rows))

        (logged,), _ = trainer.logger.experiment.log.call_args
        table = logged["hop_bands"]
        assert table.columns == ["hop_band", "hop_low", "k", "ndcg", "n_pos"]
        assert table.data == [["0-2", 0, 5, 0.5, 3.0], ["3-5", 3, 5, 0.2, 7.0]]

    def test_no_op_without_wandb_logger(self):
        rows = [{"hop_band": "0-2", "hop_low": 0, "k": 5, "ndcg": 0.5, "n_pos": 3.0}]
        trainer = types.SimpleNamespace(logger=MagicMock(spec=CSVLogger))

        HopBandTableLogger().on_test_end(trainer, self._make_module(rows))

        trainer.logger.experiment.log.assert_not_called()

    def test_no_op_when_hop_bands_disabled(self):
        trainer = types.SimpleNamespace(logger=MagicMock(spec=WandbLogger))

        HopBandTableLogger().on_test_end(trainer, self._make_module([]))

        trainer.logger.experiment.log.assert_not_called()
