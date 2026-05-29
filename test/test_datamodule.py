"""Integration smoke tests for ActivityDataModule.

Marked ``integration`` and skipped automatically when the processed dataset or config is not
available in the environment.
"""

import pytest
import torch

from activitygraphs.config import load_config
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.utils import get_project_root

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def datamodule() -> ActivityDataModule:
    root = get_project_root()
    try:
        cfg = load_config(root, verbose=False, data="toronto")
        dm = ActivityDataModule(cfg, val_size=0.2, test_size=0.1, seed=42, batch_size=64)
        dm.setup()
    except Exception as exc:  # config or processed data not present in this environment
        pytest.skip(f"toronto dataset/config unavailable: {exc}")
    return dm


class TestActivityDataModule:
    def test_split_covers_all_individuals(self, datamodule):
        # Subsets share the full (unsliced) spatial_labels tensor, so shape[0] is the total count.
        total = datamodule.train_dataset.spatial_labels.shape[0]
        n_split = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
        assert n_split == total

    def test_splits_are_disjoint(self, datamodule):
        # Regression guard for the Step 2 leak: the three splits must not share individuals.
        train = set(datamodule.train_dataset.indices())
        val = set(datamodule.val_dataset.indices())
        test = set(datamodule.test_dataset.indices())
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_pos_weight_positive_and_train_only(self, datamodule):
        assert datamodule.pos_weight > 0

        # Recompute independently over the train split only and confirm it matches.
        idx = datamodule.train_dataset.indices()
        y = datamodule.train_dataset.spatial_labels[idx]
        num_pos = y.sum()
        expected = torch.sqrt((y.numel() - num_pos) / num_pos)
        torch.testing.assert_close(datamodule.pos_weight, expected)
