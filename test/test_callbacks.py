"""Unit tests for ml callbacks (EpochMetricsCollector)."""

import types

import pytest
import torch

from activitygraphs.ml.callbacks import EpochMetricsCollector


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
