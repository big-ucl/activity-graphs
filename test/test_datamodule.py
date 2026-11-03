"""Integration smoke tests for SurveyDataModule.

Marked ``integration`` and skipped automatically when the dataset is not on disk.
"""

import pytest

from activitygraphs.datasets import HouseholdSplit, SurveyDataModule

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def datamodule(load_or_skip) -> SurveyDataModule:
    def build(cfg, root) -> SurveyDataModule:
        dm = SurveyDataModule(cfg, HouseholdSplit(val_size=0.2, test_size=0.1), seed=42, batch_size=64, num_workers=0)
        dm.prepare_data()
        dm.setup()
        return dm

    return load_or_skip("thats", build)


class TestSurveyDataModule:
    def test_split_covers_all_users(self, datamodule):
        train = datamodule.train_dataset.user_idx.tolist()
        val = datamodule.val_dataset.user_idx.tolist()
        test = datamodule.test_dataset.user_idx.tolist()
        assert sorted(train + val + test) == list(range(len(train) + len(val) + len(test)))

    def test_splits_are_disjoint(self, datamodule):
        # Regression guard for the Step 2 leak: the three splits must not share individuals.
        train = set(datamodule.train_dataset.user_idx.tolist())
        val = set(datamodule.val_dataset.user_idx.tolist())
        test = set(datamodule.test_dataset.user_idx.tolist())
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
