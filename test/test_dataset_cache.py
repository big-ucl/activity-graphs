"""Unit tests for split and scaler cache fingerprinting."""

from dataclasses import dataclass

import pytest
import torch
from omegaconf import OmegaConf

from activitygraphs.ml.dataset import dataset_fingerprint, split_indices


@dataclass
class FakeDataset:
    """Minimal stand-in exposing only what ``dataset_fingerprint`` reads."""

    spatial_features: torch.Tensor
    demographics: torch.Tensor
    num_nodes: int

    def __len__(self) -> int:
        return self.demographics.shape[0]


def make_dataset(n_users: int = 40, n_nodes: int = 12, n_demo: int = 5) -> FakeDataset:
    return FakeDataset(
        spatial_features=torch.zeros((n_users, n_nodes, 4)),
        demographics=torch.zeros((n_users, n_demo)),
        num_nodes=n_nodes,
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
