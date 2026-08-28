"""Unit tests for split and scaler cache fingerprinting."""

import json
from dataclasses import dataclass

import pytest
import torch
from omegaconf import OmegaConf

from activitygraphs.ml.dataset import dataset_fingerprint, split_indices


@dataclass
class FakeDataset:
    """Minimal stand-in exposing only what ``dataset_fingerprint`` and ``split_indices`` read."""

    spatial_features: torch.Tensor
    demographics: torch.Tensor
    num_nodes: int
    group_ids: torch.Tensor

    def __len__(self) -> int:
        return self.demographics.shape[0]


def make_dataset(n_users: int = 40, n_nodes: int = 12, n_demo: int = 5, users_per_group: int = 1) -> FakeDataset:
    """Build a fake dataset whose users are laid out in consecutive groups of ``users_per_group``.

    ``users_per_group=1`` reproduces the ungrouped case (Geneva, where ``hh_id`` is ``user_id``).
    """
    group_ids = torch.arange(n_users) // users_per_group

    return FakeDataset(
        spatial_features=torch.zeros((n_users, n_nodes, 4)),
        demographics=torch.zeros((n_users, n_demo)),
        num_nodes=n_nodes,
        group_ids=group_ids,
    )


def make_cfg(name: str = "GenevaTPG", epochs: int = 50):
    return OmegaConf.create({
        "data": {
            "name": name,
            "paths": {"processed": "data/processed/${..name}", "pyg_datasets": "${..paths.processed}/PyGDatasets"},
        },
        "train": {"epochs": epochs, "split_seed": 42, "loss": {"type": "bpr"}},
    })


@pytest.fixture
def cfg():
    return make_cfg()


class TestDatasetFingerprint:
    def test_is_stable_across_calls(self, cfg):
        dataset = make_dataset()
        assert dataset_fingerprint(cfg, dataset) == dataset_fingerprint(cfg, dataset)

    def test_detects_a_changed_user_count(self, cfg):
        assert dataset_fingerprint(cfg, make_dataset(n_users=40)) != dataset_fingerprint(cfg, make_dataset(n_users=41))

    def test_detects_a_changed_node_universe(self, cfg):
        assert dataset_fingerprint(cfg, make_dataset(n_nodes=12)) != dataset_fingerprint(
            cfg, make_dataset(n_nodes=2448)
        )

    def test_detects_a_changed_demographic_width(self, cfg):
        # The case that would otherwise surface as an sklearn feature-count mismatch inside apply_scalers.
        assert dataset_fingerprint(cfg, make_dataset(n_demo=5)) != dataset_fingerprint(cfg, make_dataset(n_demo=9))

    def test_detects_a_changed_data_config(self):
        dataset = make_dataset()
        assert dataset_fingerprint(make_cfg(name="cmap"), dataset) != dataset_fingerprint(make_cfg(), dataset)

    def test_ignores_training_hyperparameters(self):
        # Only cfg.data is hashed: an epochs/loss override must not invalidate the caches.
        dataset = make_dataset()
        assert dataset_fingerprint(make_cfg(epochs=500), dataset) == dataset_fingerprint(make_cfg(epochs=50), dataset)

    def test_resolves_interpolations(self, cfg):
        # Renaming the dataset moves every interpolated path, so the digest must not hash a raw "${..name}".
        assert "${" not in OmegaConf.to_yaml(cfg.data, resolve=True)


class TestSplitIndicesCache:
    def test_reuses_a_matching_cache(self, tmp_path, cfg):
        dataset = make_dataset()
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        fingerprint = dataset_fingerprint(cfg, dataset)

        first = split_indices(dataset, 0.2, 0.1, 42, fingerprint, cache_path=cache)
        second = split_indices(dataset, 0.2, 0.1, 42, fingerprint, cache_path=cache)

        assert first == second
        assert cache.exists()

    def test_re_splits_when_the_dataset_changed(self, tmp_path, cfg):
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        old = make_dataset(n_users=40)
        split_indices(old, 0.2, 0.1, 42, dataset_fingerprint(cfg, old), cache_path=cache)

        new = make_dataset(n_users=60)
        train, val, test = split_indices(new, 0.2, 0.1, 42, dataset_fingerprint(cfg, new), cache_path=cache)

        assert sorted(train + val + test) == list(range(60))

    def test_rebuilt_cache_is_reused_on_the_next_load(self, tmp_path, cfg):
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        old = make_dataset(n_users=40)
        split_indices(old, 0.2, 0.1, 42, dataset_fingerprint(cfg, old), cache_path=cache)

        new = make_dataset(n_users=60)
        new_fingerprint = dataset_fingerprint(cfg, new)
        rebuilt = split_indices(new, 0.2, 0.1, 42, new_fingerprint, cache_path=cache)

        assert split_indices(new, 0.2, 0.1, 42, new_fingerprint, cache_path=cache) == rebuilt

    def test_split_is_unchanged_by_fingerprinting(self, tmp_path, cfg):
        # The existing caches are deleted when this lands; the rebuilt split must reproduce the old one.
        dataset = make_dataset()
        fingerprint = dataset_fingerprint(cfg, dataset)
        cached = split_indices(dataset, 0.2, 0.1, 42, fingerprint, cache_path=tmp_path / "splits.json")
        uncached = split_indices(dataset, 0.2, 0.1, 42, fingerprint)

        assert cached == uncached

    def test_detects_a_changed_grouping(self, cfg):
        # An ungrouped cache must not be silently reused for a grouped run, and vice versa.
        assert dataset_fingerprint(cfg, make_dataset(users_per_group=1)) != dataset_fingerprint(
            cfg, make_dataset(users_per_group=2)
        )


class TestGroupedSplit:
    @pytest.mark.parametrize("users_per_group", [1, 2, 5])
    def test_splits_partition_every_user(self, users_per_group, cfg):
        dataset = make_dataset(n_users=60, users_per_group=users_per_group)
        train, val, test = split_indices(dataset, 0.2, 0.1, 42, dataset_fingerprint(cfg, dataset))

        assert sorted(train + val + test) == list(range(60))

    @pytest.mark.parametrize("users_per_group", [2, 5])
    def test_no_group_spans_two_splits(self, users_per_group, cfg):
        # The leak the household-grouped split exists to prevent: co-residents share a home tract
        # and overlapping visit sets, so a household in both train and test leaks home conditioning.
        dataset = make_dataset(n_users=60, users_per_group=users_per_group)
        train, val, test = split_indices(dataset, 0.2, 0.1, 42, dataset_fingerprint(cfg, dataset))

        groups = {
            name: set(dataset.group_ids[idx].tolist())
            for name, idx in zip(("train", "val", "test"), (train, val, test))
        }

        assert groups["train"] & groups["val"] == set()
        assert groups["train"] & groups["test"] == set()
        assert groups["val"] & groups["test"] == set()

    def test_split_is_deterministic_for_a_seed(self, cfg):
        # Results are reported as mean +- sd over train seeds, so the split must be a function of
        # the split seed alone, not of the order group ids happened to be built in.
        dataset = make_dataset(n_users=60, users_per_group=2)
        fingerprint = dataset_fingerprint(cfg, dataset)

        assert split_indices(dataset, 0.2, 0.1, 42, fingerprint) == split_indices(dataset, 0.2, 0.1, 42, fingerprint)

    def test_different_seeds_give_different_splits(self, cfg):
        dataset = make_dataset(n_users=60, users_per_group=2)
        fingerprint = dataset_fingerprint(cfg, dataset)

        assert split_indices(dataset, 0.2, 0.1, 42, fingerprint) != split_indices(dataset, 0.2, 0.1, 7, fingerprint)

    def test_splits_are_roughly_the_requested_sizes(self, cfg):
        # Grouping quantises the split, so only assert the sizes are in the right neighbourhood.
        dataset = make_dataset(n_users=600, users_per_group=2)
        train, val, test = split_indices(dataset, 0.2, 0.1, 42, dataset_fingerprint(cfg, dataset))

        assert len(test) == pytest.approx(60, abs=10)
        assert len(val) == pytest.approx(120, abs=10)
        assert len(train) == pytest.approx(420, abs=20)

    def test_indices_are_json_serialisable(self, tmp_path, cfg):
        # GroupShuffleSplit returns numpy arrays; the cache write must not choke on them.
        dataset = make_dataset(n_users=60, users_per_group=2)
        cache = tmp_path / "splits.json"
        split_indices(dataset, 0.2, 0.1, 42, dataset_fingerprint(cfg, dataset), cache_path=cache)

        assert json.loads(cache.read_text())["train"]
