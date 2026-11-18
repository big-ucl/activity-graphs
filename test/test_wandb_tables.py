"""Unit tests for the band table uploaded to W&B by ``report-results``."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from activitygraphs.wandb_tables import _wandb_group, band_seed_means


def per_user_band_recall() -> pl.DataFrame:
    return pl.DataFrame({
        "name": ["MLP", "MLP", "MLP", "Gravity"],
        "seed": [42, 42, 42, None],
        "user_idx": [0, 1, 0, 0],
        "band": ["0-2 km", "0-2 km", "2-5 km", "0-2 km"],
        "band_lo": [0.0, 0.0, 2.0, 0.0],
        "band_recall": [0.2, 0.4, 0.5, 0.1],
    })


def per_user_band_auc() -> pl.DataFrame:
    return pl.DataFrame({
        "name": ["MLP", "MLP", "Gravity"],
        "seed": [42, 42, None],
        "user_idx": [0, 1, 0],
        "band": ["0-2 km", "0-2 km", "0-2 km"],
        "band_lo": [0.0, 0.0, 0.0],
        "auc": [0.6, 0.8, 0.5],
    })


def test_the_band_means_average_the_users_of_each_model_seed_and_band():
    band_means = band_seed_means(per_user_band_recall(), per_user_band_auc())

    expected_means = pl.DataFrame({
        "model": ["Gravity", "MLP", "MLP"],
        "seed": [None, 42, 42],
        "band": ["0-2 km", "0-2 km", "2-5 km"],
        "band_lo": [0.0, 0.0, 2.0],
        "band_recall": [0.1, 0.3, 0.5],
        "auc": [0.5, 0.7, None],
    })
    assert_frame_equal(band_means, expected_means)


def test_without_score_vectors_the_auc_is_null():
    band_means = band_seed_means(per_user_band_recall(), None)

    assert band_means["auc"].null_count() == band_means.height


def test_the_group_is_the_one_of_the_training_runs():
    assert _wandb_group("TTS2016-21-20261117-183417", "edge_ablation") == "TTS2016-edge_ablation-20261117-183417"


def test_an_unnumbered_run_directory_raises():
    with pytest.raises(ValueError, match="numbered run directory"):
        _wandb_group("TPG-20261117-175704", "edge_ablation")
