"""Integration smoke tests for ActivityDataModule.

Marked ``integration`` and skipped automatically when the dataset is not on disk.
"""

import pytest

from activitygraphs.ml.datamodule import ActivityDataModule

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def datamodule(load_or_skip) -> ActivityDataModule:
    def build(cfg, root) -> ActivityDataModule:
        dm = ActivityDataModule(cfg, val_size=0.2, test_size=0.1, seed=42, batch_size=64)
        dm.setup()
        return dm

    return load_or_skip("thats", build)


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
