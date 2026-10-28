"""Unit tests for split and transform cache fingerprinting."""

import json
from dataclasses import dataclass

import polars as pl
import pytest
import torch
import torch_geometric as pyg
from omegaconf import OmegaConf

from activitygraphs.datasets.dataset import (
    Fingerprint,
    ProcessedFile,
    dataset_fingerprint,
    read_fingerprint,
    tensors_fingerprint,
)
from activitygraphs.datasets.splits import HouseholdSplit, load_split_indices, read_split_indices
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, TensorSchema
from activitygraphs.tensors import SurveyTensors


@dataclass
class FakeDataset:
    """Minimal stand-in exposing only what ``load_split_indices`` reads, beside the tensors it stands for."""

    tensors: SurveyTensors

    @property
    def group_ids(self) -> torch.Tensor:
        return self.tensors.group_ids

    def __len__(self) -> int:
        return self.tensors.user_attributes.shape[0]


def make_group(group: str, width: int, kind: ColumnKind = ColumnKind.CONTINUOUS) -> TensorSchema:
    return TensorSchema(group, tuple(ColumnSpec(f"{group}_{i}", kind) for i in range(width)))


def make_schema(n_user_node: int, n_demo: int, demo_kind: ColumnKind) -> DatasetSchema:
    return DatasetSchema(
        locations=make_group("locations", 0),
        edges=make_group("edges", 0),
        user_nodes=make_group("user_nodes", n_user_node),
        user_attributes=make_group("user_attributes", n_demo, demo_kind),
        distances=make_group("distances", 1),
        targets=make_group("targets", 1),
    )


def make_dataset(
    n_users: int = 40,
    n_nodes: int = 12,
    n_demo: int = 5,
    users_per_group: int = 1,
    user_id_prefix: str = "u",
    demo_kind: ColumnKind = ColumnKind.CONTINUOUS,
) -> FakeDataset:
    """Build a fake dataset whose users are laid out in consecutive groups of ``users_per_group``.

    ``users_per_group=1`` reproduces the ungrouped case (TPG, where ``hh_id`` is ``user_id``).
    """
    group_ids = torch.arange(n_users) // users_per_group
    user_ids = pl.DataFrame({
        "user_id": [f"{user_id_prefix}{i}" for i in range(n_users)],
        "hh_id": [f"h{g}" for g in group_ids.tolist()],
    })

    return FakeDataset(
        SurveyTensors(
            pyg_graph=pyg.data.Data(num_nodes=n_nodes),
            user_node_features=torch.zeros((n_users, n_nodes, 4)),
            visit_labels=torch.zeros((n_users, n_nodes, 1)),
            user_attributes=torch.zeros((n_users, n_demo)),
            distances=torch.zeros((n_users, n_nodes, 1)),
            group_ids=group_ids,
            user_ids=user_ids,
            node_ids=pl.DataFrame({"loc_id": [f"z{i}" for i in range(n_nodes)]}),
            schema=make_schema(4, n_demo, demo_kind),
        )
    )


def fingerprint_of(cfg, dataset: FakeDataset) -> Fingerprint:
    return dataset_fingerprint(cfg, tensors_fingerprint(dataset.tensors))


def make_cfg(name: str = "TPG", epochs: int = 50):
    return OmegaConf.create({
        "data": {
            "name": name,
            "paths": {"processed": "data/processed/${..name}", "tensors": "${..paths.processed}/Tensors"},
        },
        "train": {"epochs": epochs, "split_seed": 42, "loss": {"type": "bpr"}},
    })


SPLIT = HouseholdSplit(val_size=0.2, test_size=0.1)


@pytest.fixture
def cfg():
    return make_cfg()


class TestDatasetFingerprint:
    def test_is_stable_across_calls(self, cfg):
        dataset = make_dataset()
        assert fingerprint_of(cfg, dataset) == fingerprint_of(cfg, dataset)

    def test_detects_a_changed_user_count(self, cfg):
        assert fingerprint_of(cfg, make_dataset(n_users=40)) != fingerprint_of(cfg, make_dataset(n_users=41))

    def test_detects_a_changed_node_universe(self, cfg):
        assert fingerprint_of(cfg, make_dataset(n_nodes=12)) != fingerprint_of(cfg, make_dataset(n_nodes=2448))

    def test_detects_a_changed_demographic_width(self, cfg):
        # The case that would otherwise surface as an sklearn feature-count mismatch inside FeatureTransforms.apply.
        assert fingerprint_of(cfg, make_dataset(n_demo=5)) != fingerprint_of(cfg, make_dataset(n_demo=9))

    def test_detects_a_changed_data_config(self):
        dataset = make_dataset()
        assert fingerprint_of(make_cfg(name="cmap"), dataset) != fingerprint_of(make_cfg(), dataset)

    def test_ignores_training_hyperparameters(self):
        # Only cfg.data is hashed: an epochs/loss override must not invalidate the caches.
        dataset = make_dataset()
        assert fingerprint_of(make_cfg(epochs=500), dataset) == fingerprint_of(make_cfg(epochs=50), dataset)

    def test_detects_changed_ids_at_equal_counts(self, cfg):
        # A cohort change that keeps every count must still invalidate the cached row indices.
        assert fingerprint_of(cfg, make_dataset(user_id_prefix="u")) != fingerprint_of(
            cfg, make_dataset(user_id_prefix="v")
        )

    def test_detects_a_changed_column_kind_at_equal_widths(self, cfg):
        assert fingerprint_of(cfg, make_dataset(demo_kind=ColumnKind.CONTINUOUS)) != fingerprint_of(
            cfg, make_dataset(demo_kind=ColumnKind.BINARY)
        )

    def test_is_read_back_from_the_file_beside_the_tensors(self, tmp_path, cfg):
        dataset = make_dataset()
        processed = tmp_path / cfg.data.paths.tensors / "processed"
        processed.mkdir(parents=True)
        (processed / ProcessedFile.FINGERPRINT).write_text(json.dumps(tensors_fingerprint(dataset.tensors)))

        assert read_fingerprint(cfg, tmp_path) == fingerprint_of(cfg, dataset)

    def test_reading_an_uncached_fingerprint_raises(self, tmp_path, cfg):
        with pytest.raises(FileNotFoundError):
            read_fingerprint(cfg, tmp_path)

    def test_resolves_interpolations(self, cfg):
        # Renaming the dataset moves every interpolated path, so the digest must not hash a raw "${..name}".
        assert "${" not in OmegaConf.to_yaml(cfg.data, resolve=True)


class TestSplitIndicesCache:
    def test_reuses_a_matching_cache(self, tmp_path, cfg):
        dataset = make_dataset()
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        fingerprint = fingerprint_of(cfg, dataset)

        first = load_split_indices(dataset, SPLIT, 42, fingerprint, cache_path=cache)
        second = load_split_indices(dataset, SPLIT, 42, fingerprint, cache_path=cache)

        assert first == second
        assert cache.exists()

    def test_re_splits_when_the_dataset_changed(self, tmp_path, cfg):
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        old = make_dataset(n_users=40)
        load_split_indices(old, SPLIT, 42, fingerprint_of(cfg, old), cache_path=cache)

        new = make_dataset(n_users=60)
        train, val, test = load_split_indices(new, SPLIT, 42, fingerprint_of(cfg, new), cache_path=cache)

        assert sorted(train + val + test) == list(range(60))

    def test_rebuilt_cache_is_reused_on_the_next_load(self, tmp_path, cfg):
        cache = tmp_path / "splits_42_val0.2_test0.1.json"
        old = make_dataset(n_users=40)
        load_split_indices(old, SPLIT, 42, fingerprint_of(cfg, old), cache_path=cache)

        new = make_dataset(n_users=60)
        new_fingerprint = fingerprint_of(cfg, new)
        rebuilt = load_split_indices(new, SPLIT, 42, new_fingerprint, cache_path=cache)

        assert load_split_indices(new, SPLIT, 42, new_fingerprint, cache_path=cache) == rebuilt

    def test_split_is_unchanged_by_fingerprinting(self, tmp_path, cfg):
        # The existing caches are deleted when this lands; the rebuilt split must reproduce the old one.
        dataset = make_dataset()
        fingerprint = fingerprint_of(cfg, dataset)
        cached = load_split_indices(dataset, SPLIT, 42, fingerprint, cache_path=tmp_path / "splits.json")
        uncached = load_split_indices(dataset, SPLIT, 42, fingerprint)

        assert cached == uncached

    def test_detects_a_changed_grouping(self, cfg):
        # An ungrouped cache must not be silently reused for a grouped run, and vice versa.
        assert fingerprint_of(cfg, make_dataset(users_per_group=1)) != fingerprint_of(
            cfg, make_dataset(users_per_group=2)
        )


class TestReadSplitIndices:
    def test_reads_the_split_load_split_indices_cached(self, tmp_path, cfg):
        dataset = make_dataset()
        cache = tmp_path / "splits.json"
        fingerprint = fingerprint_of(cfg, dataset)
        cached = load_split_indices(dataset, SPLIT, 42, fingerprint, cache_path=cache)

        assert read_split_indices(cache, fingerprint) == cached

    def test_raises_when_the_dataset_changed(self, tmp_path, cfg):
        cache = tmp_path / "splits.json"
        old = make_dataset(n_users=40)
        load_split_indices(old, SPLIT, 42, fingerprint_of(cfg, old), cache_path=cache)

        with pytest.raises(ValueError, match="fingerprint"):
            read_split_indices(cache, fingerprint_of(cfg, make_dataset(n_users=60)))

    def test_raises_when_nothing_is_cached(self, tmp_path, cfg):
        with pytest.raises(FileNotFoundError):
            read_split_indices(tmp_path / "splits.json", fingerprint_of(cfg, make_dataset()))


class TestGroupedSplit:
    @pytest.mark.parametrize("users_per_group", [1, 2, 5])
    def test_splits_partition_every_user(self, users_per_group, cfg):
        dataset = make_dataset(n_users=60, users_per_group=users_per_group)
        train, val, test = load_split_indices(dataset, SPLIT, 42, fingerprint_of(cfg, dataset))

        assert sorted(train + val + test) == list(range(60))

    @pytest.mark.parametrize("users_per_group", [2, 5])
    def test_no_group_spans_two_splits(self, users_per_group, cfg):
        # The leak the household-grouped split exists to prevent: co-residents share a home tract
        # and overlapping visit sets, so a household in both train and test leaks home conditioning.
        dataset = make_dataset(n_users=60, users_per_group=users_per_group)
        train, val, test = load_split_indices(dataset, SPLIT, 42, fingerprint_of(cfg, dataset))

        groups = {
            name: set(dataset.group_ids[idx].tolist())
            for name, idx in zip(("train", "val", "test"), (train, val, test), strict=True)
        }

        assert groups["train"] & groups["val"] == set()
        assert groups["train"] & groups["test"] == set()
        assert groups["val"] & groups["test"] == set()

    def test_split_is_deterministic_for_a_seed(self, cfg):
        # Results are reported as mean +- sd over train seeds, so the split must be a function of
        # the split seed alone, not of the order group ids happened to be built in.
        dataset = make_dataset(n_users=60, users_per_group=2)
        fingerprint = fingerprint_of(cfg, dataset)

        assert load_split_indices(dataset, SPLIT, 42, fingerprint) == load_split_indices(
            dataset, SPLIT, 42, fingerprint
        )

    def test_different_seeds_give_different_splits(self, cfg):
        dataset = make_dataset(n_users=60, users_per_group=2)
        fingerprint = fingerprint_of(cfg, dataset)

        assert load_split_indices(dataset, SPLIT, 42, fingerprint) != load_split_indices(dataset, SPLIT, 7, fingerprint)

    def test_splits_are_roughly_the_requested_sizes(self, cfg):
        # Grouping quantises the split, so only assert the sizes are in the right neighbourhood.
        dataset = make_dataset(n_users=600, users_per_group=2)
        train, val, test = load_split_indices(dataset, SPLIT, 42, fingerprint_of(cfg, dataset))

        assert len(test) == pytest.approx(60, abs=10)
        assert len(val) == pytest.approx(120, abs=10)
        assert len(train) == pytest.approx(420, abs=20)

    def test_indices_are_json_serialisable(self, tmp_path, cfg):
        # GroupShuffleSplit returns numpy arrays; the cache write must not choke on them.
        dataset = make_dataset(n_users=60, users_per_group=2)
        cache = tmp_path / "splits.json"
        load_split_indices(dataset, SPLIT, 42, fingerprint_of(cfg, dataset), cache_path=cache)

        assert json.loads(cache.read_text())["train"]
