"""Snapshot of the model inputs, end to end from a synthetic survey to the ``FeatureEncoder``.

The snapshot pins the inputs a model sees for every combination of input flags, so a refactor of the
dataset, scaling or feature-assembly path can be checked to leave them unchanged. Regenerate it with
``uv run python test/test_input_snapshot.py`` only when a change to the inputs is intended.
"""

import dataclasses
import datetime as dt
import itertools
import json
from pathlib import Path

import city2graph as c2g
import geopandas as gpd
import numpy as np
import polars as pl
import pytest
import torch
import torch_geometric as pyg
from shapely.geometry import box

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, USER_JOURNEY_SCHEMA, USER_SCHEMA, Mode, Purpose
from activitygraphs.config import HomePEConfig, read_config
from activitygraphs.datasets import HouseholdSplit, datamodule
from activitygraphs.datasets.datamodule import positional_encodings
from activitygraphs.datasets.dataset import ProcessedFile, VisitDataset, load_dataset
from activitygraphs.datasets.transforms import fit_transforms
from activitygraphs.experiments.variants import Variant, build_feature_encoder
from activitygraphs.ml.schema_encoder import default_emb_dim
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.surveys.cmap import CMAP_DEMOGRAPHICS_SCHEMA, CMAPSurvey
from activitygraphs.tensors import SurveyTensors, build_tensors
from activitygraphs.utils import add_lon_lat_from_centroid, get_project_root

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "model_inputs.pt"

GRID_SIDE = 3
NUM_HOUSEHOLDS = 8
USERS_PER_HOUSEHOLD = 2
NUM_USERS = NUM_HOUSEHOLDS * USERS_PER_HOUSEHOLD
TRAIN_IDX = list(range(10))
VISIT_PURPOSES = [Purpose.WORK_MAIN, Purpose.SHOP, Purpose.LEISURE_OTHER, Purpose.STUDY]
FLAGS = ("full_info", "use_demographics", "use_home_pe", "use_rwpe")
SURVEYS = ("cmap", "no_demographics")
HOME_PE = HomePEConfig(bins=8, max_hops=40)


class NoDemographicsSurvey(CMAPSurvey):
    """A ``CMAPSurvey`` without demographic columns, as TPG has none."""

    @property
    def demographics_schema(self) -> TensorSchema:
        """An empty schema."""
        return TensorSchema("user_attributes", ())


SURVEY_CLASSES = {
    "cmap": CMAPSurvey,
    "no_demographics": NoDemographicsSurvey,
}


def build_zones() -> gpd.GeoDataFrame:
    """Return a ``GRID_SIDE x GRID_SIDE`` grid of square zones near Chicago, as survey locations."""
    cells = list(itertools.product(range(GRID_SIDE), range(GRID_SIDE)))
    geometry = [box(-87.9 + 0.01 * i, 41.7 + 0.01 * j, -87.9 + 0.01 * (i + 1), 41.7 + 0.01 * (j + 1)) for i, j in cells]
    loc_ids = [f"z{k:02d}" for k in range(len(cells))]
    zones = gpd.GeoDataFrame({"loc_id": loc_ids, "loc_name": loc_ids, "type": "zone"}, geometry=geometry, crs=CRS)

    return add_lon_lat_from_centroid(zones, "loc_id")[LOCATIONS_COLUMNS]


def build_graph(zones: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return the contiguity graph over ``zones`` with density and land-use columns, built as ``graph/build.py`` does."""
    rng = np.random.default_rng(0)
    projected = zones.to_crs(zones.estimate_utm_crs())
    projected["pop_density"] = rng.lognormal(-6.0, 1.0, len(zones))
    projected["poi_shop"] = rng.lognormal(-9.0, 1.5, len(zones))
    projected["land_use_residential"] = rng.uniform(0.0, 1.0, len(zones))
    nodes_gdf, edges_gdf = c2g.contiguity_graph(projected.set_index("loc_id"), set_point_nodes=True)

    # city2graph's edge order depends on the hash seed.
    return nodes_gdf.to_crs(CRS), edges_gdf.sort_index().to_crs(CRS)


def build_users(loc_ids: list[str]) -> pl.DataFrame:
    """Return ``NUM_USERS`` users in households of ``USERS_PER_HOUSEHOLD`` with CMAP demographics."""
    rng = np.random.default_rng(1)
    hh = np.repeat(np.arange(NUM_HOUSEHOLDS), USERS_PER_HOUSEHOLD)
    hh_home = rng.choice(loc_ids, NUM_HOUSEHOLDS)
    hh_adults = rng.integers(1, 4, NUM_HOUSEHOLDS)
    hh_children = rng.integers(0, 3, NUM_HOUSEHOLDS)
    hh_vehicles = rng.integers(0, 3, NUM_HOUSEHOLDS)
    hh_bicycle = rng.integers(0, 4, NUM_HOUSEHOLDS)
    hh_income = rng.choice([1, 5, 7, 8, 9, 10], NUM_HOUSEHOLDS)
    has_missing = rng.random(NUM_USERS) < 0.2
    employment = rng.choice([1, 2, 5, 6, 11], NUM_USERS)

    users = pl.DataFrame({
        "user_id": [f"u{k:02d}" for k in range(NUM_USERS)],
        "hh_id": [f"h{h:02d}" for h in hh],
        "home_loc_id": hh_home[hh],
        "has_missing_demographics": has_missing,
        "hh_num_adults": hh_adults[hh],
        "hh_num_children": hh_children[hh],
        "hh_num_vehicles": hh_vehicles[hh],
        "hh_num_bicycle": hh_bicycle[hh],
        "income_band": hh_income[hh],
        "age_band": rng.integers(5, 13, NUM_USERS),
        "employment_status": [
            None if missing else int(code) for missing, code in zip(has_missing, employment, strict=True)
        ],
        "student_status": rng.choice([0, 2], NUM_USERS),
        "can_drive": rng.random(NUM_USERS) < 0.7,
        "has_transit_pass": rng.random(NUM_USERS) < 0.3,
    })

    return users.cast(USER_SCHEMA)


def build_journeys(users: pl.DataFrame, loc_ids: list[str]) -> pl.DataFrame:
    """Return single-leg home-anchored round trips to two or three destinations per user."""
    rng = np.random.default_rng(2)
    rows = []
    for user_id, home in users.select("user_id", "home_loc_id").iter_rows():
        destinations = rng.choice([loc for loc in loc_ids if loc != home], rng.integers(2, 4), replace=False)
        for j, destination in enumerate(destinations):
            purpose = VISIT_PURPOSES[rng.integers(len(VISIT_PURPOSES))]
            for leg, (dep, arr, dep_purpose, arr_purpose) in enumerate([
                (home, destination, Purpose.HOME, purpose),
                (destination, home, purpose, Purpose.HOME),
            ]):
                rows.append({
                    "user_id": user_id,
                    "journey_id": f"{user_id}-{j}-{leg}",
                    "leg_id": 0,
                    "leg_mode": Mode.CAR,
                    "leg_line": None,
                    "duration": dt.timedelta(minutes=15),
                    "dep_day": dt.date(2024, 5, 13),
                    "dep_time": dt.time(8 + 2 * j + leg, 0),
                    "dep_purpose": dep_purpose,
                    "dep_loc_id": dep,
                    "arr_loc_id": arr,
                    "arr_purpose": arr_purpose,
                })

    return pl.DataFrame(rows, schema=USER_JOURNEY_SCHEMA)


def build_survey_tensors(survey_name: str) -> SurveyTensors:
    """Return the tensors of the synthetic survey ``survey_name``."""
    zones = build_zones()
    nodes_gdf, edges_gdf = build_graph(zones)
    loc_ids = zones["loc_id"].tolist()
    users = build_users(loc_ids)
    survey_cls = SURVEY_CLASSES[survey_name]
    survey = survey_cls(zones, build_journeys(users, loc_ids), users).with_filter("zone")

    return build_tensors(survey, nodes_gdf, edges_gdf)


def build_dataset(survey_name: str, root: Path) -> VisitDataset:
    """Return the unscaled ``VisitDataset`` of the synthetic survey ``survey_name``, processed under ``root``."""
    tensors = build_survey_tensors(survey_name)

    return VisitDataset(str(root), lambda: tensors, pre_transform=positional_encodings())


def build_model_inputs(survey_name: str, root: Path) -> dict[str, torch.Tensor]:
    """Run the synthetic survey through the dataset, scaling and feature assembly.

    Args:
        survey_name: One of ``SURVEYS``.
        root: Directory for the processed tensors.

    Returns:
        The scaled edge attributes, the labels, and the ``FeatureEncoder`` output over all users for every
        combination of ``FLAGS``, keyed by the active flags joined with ``+`` (``"none"`` for no flags).
    """
    dataset = build_dataset(survey_name, root)
    fit_transforms(dataset, TRAIN_IDX, fingerprint={}).apply(dataset)

    batch = next(iter(pyg.loader.DataLoader(dataset, batch_size=len(dataset))))
    mlp = read_config(get_project_root(), verbose=False).models.mlp

    inputs = {"edge_attr": batch.edge_attr, "y": batch.y}
    for active in itertools.product([False, True], repeat=len(FLAGS)):
        flags = dict(zip(FLAGS, active, strict=True))
        key = "+".join(name for name, on in flags.items() if on) or "none"
        torch.manual_seed(0)
        encoder = build_feature_encoder(Variant(mlp, **flags), dataset, HOME_PE)
        with torch.no_grad():
            inputs[key] = encoder(batch)
        assert inputs[key].shape[1] == encoder.out_dim

    return inputs


@pytest.fixture(scope="module")
def snapshot() -> dict[str, dict[str, torch.Tensor]]:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(f"No snapshot at {SNAPSHOT_PATH}; generate it with `uv run python {Path(__file__).name}`.")

    return torch.load(SNAPSHOT_PATH, weights_only=True)


@pytest.mark.parametrize("survey_name", SURVEYS)
def test_model_inputs_match_the_snapshot(survey_name, snapshot, tmp_path):
    expected = snapshot[survey_name]
    actual = build_model_inputs(survey_name, tmp_path)

    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor, msg=lambda m, key=key: f"{key}: {m}")


def test_the_shared_graph_is_undirected(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    _, edges_gdf = build_graph(build_zones())

    assert pyg.utils.is_undirected(dataset.edge_index, num_nodes=dataset.num_nodes)
    assert dataset.edge_index.size(1) == 2 * len(edges_gdf)
    assert dataset.pyg_graph.random_walk_pe.abs().sum() > 0


def test_the_cache_carries_the_survey_ids_and_the_fingerprint(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    zones = build_zones()
    users = build_users(zones["loc_id"].tolist())

    assert dataset.user_ids.equals(users.sort("user_id").select("user_id", "hh_id"))
    assert dataset.node_ids["loc_id"].to_list() == zones["loc_id"].tolist()

    fingerprint = json.loads((Path(dataset.processed_dir) / ProcessedFile.FINGERPRINT).read_text())
    assert fingerprint["n_users"] == NUM_USERS
    assert fingerprint["n_groups"] == NUM_HOUSEHOLDS


def test_an_item_carries_the_location_and_user_node_features_apart(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    item = dataset[3]
    batch = next(iter(pyg.loader.DataLoader(dataset, batch_size=len(dataset))))

    assert "x" not in item
    torch.testing.assert_close(item.x_loc, dataset.location_x)
    torch.testing.assert_close(item.x_user_node, dataset.user_node_features[3])
    assert item.num_nodes == dataset.num_nodes
    assert batch.num_nodes == NUM_USERS * dataset.num_nodes


def test_an_item_carries_the_distances_in_metres_beside_the_scaled_ones(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    metres = dataset.distances.clone()
    transforms = fit_transforms(dataset, TRAIN_IDX, fingerprint={})
    transforms.apply(dataset)
    item = dataset[3]

    scaled = transforms.groups["distances"].apply(metres[3])
    torch.testing.assert_close(item.distances_m, metres[3])
    torch.testing.assert_close(item.distances, scaled)


@pytest.mark.parametrize(
    ("survey_name", "user_attributes"),
    [("cmap", CMAP_DEMOGRAPHICS_SCHEMA), ("no_demographics", TensorSchema("user_attributes", ()))],
)
def test_the_cache_carries_the_schema_of_every_tensor(survey_name, user_attributes, tmp_path):
    dataset = build_dataset(survey_name, tmp_path)
    schema = dataset.schema

    assert schema.user_attributes == user_attributes
    assert schema.locations.names == ["pop_density", "poi_shop", "land_use_residential"]
    assert schema.locations.indices(ColumnKind.CONTINUOUS) == [0, 1, 2]
    assert schema.locations.width == dataset.location_x.shape[1]
    assert schema.edges.width == dataset.edge_attr.shape[1]
    assert schema.user_nodes.width == dataset.user_node_features.shape[-1]
    assert schema.user_attributes.numeric.width == dataset.user_attributes.shape[-1]
    assert schema.user_attributes.categorical.width == dataset.user_attributes_cat.shape[-1]
    assert schema.distances.width == dataset.distances.shape[-1]
    assert schema.targets.width == dataset.visit_labels.shape[-1]


def test_without_demographics_use_demographics_leaves_the_inputs_unchanged(tmp_path):
    inputs = build_model_inputs("no_demographics", tmp_path)

    for key, tensor in inputs.items():
        if "use_demographics" in key:
            without = key.replace("use_demographics", "").replace("++", "+").strip("+") or "none"
            torch.testing.assert_close(tensor, inputs[without], rtol=0, atol=0, msg=lambda m, key=key: f"{key}: {m}")


def test_categorical_attributes_are_cached_as_codes_and_embedded(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    users = build_users(build_zones()["loc_id"].tolist()).sort("user_id")
    employment = users["employment_status"].to_list()
    categorical = dataset.schema.user_attributes.categorical
    employment_idx = categorical.index_of("employment_status")
    categories = categorical.columns[employment_idx].categories
    expected_codes = torch.tensor([0 if code is None else categories.index(str(code)) + 1 for code in employment])

    fit_transforms(dataset, TRAIN_IDX, fingerprint={}).apply(dataset)
    batch = next(iter(pyg.loader.DataLoader(dataset, batch_size=len(dataset))))
    mlp = read_config(get_project_root(), verbose=False).models.mlp
    plain = build_feature_encoder(Variant(mlp, use_demographics=False), dataset, HOME_PE)
    encoder = build_feature_encoder(Variant(mlp), dataset, HOME_PE)
    embedded_width = 0
    for spec in categorical.columns:
        embedded_width += default_emb_dim(spec.cardinality)

    assert None in employment
    assert categorical.names == ["age_band", "employment_status", "student_status", "has_transit_pass", "income_band"]
    assert dataset.user_attributes.shape[-1] == CMAP_DEMOGRAPHICS_SCHEMA.numeric.width
    torch.testing.assert_close(dataset.user_attributes_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(batch.user_attr_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    assert encoder.out_dim == plain.out_dim + CMAP_DEMOGRAPHICS_SCHEMA.numeric.width + embedded_width
    assert encoder(batch).shape == (batch.num_nodes, encoder.out_dim)


def test_a_tensor_whose_width_differs_from_its_schema_raises():
    tensors = build_survey_tensors("cmap")
    extra_column = ColumnSpec("extra", ColumnKind.BINARY)
    user_nodes = TensorSchema("user_nodes", (*tensors.schema.user_nodes.columns, extra_column))
    schema = dataclasses.replace(tensors.schema, user_nodes=user_nodes)

    with pytest.raises(ValueError, match=r"`user_nodes` tensors have \(numeric, categorical\) widths \(1, 0\)"):
        dataclasses.replace(tensors, schema=schema)


def test_a_user_without_exactly_one_home_node_raises():
    tensors = build_survey_tensors("cmap")
    is_home_idx = tensors.schema.user_nodes.index_of("is_home")
    two_homes = tensors.user_node_features.clone()
    two_homes[0, :2, is_home_idx] = 1.0

    with pytest.raises(ValueError, match="1 users do not have exactly one home node"):
        dataclasses.replace(tensors, user_node_features=two_homes)


def test_a_graph_without_an_edge_index_raises():
    tensors = build_survey_tensors("cmap")
    graph = tensors.pyg_graph.clone()
    del graph.edge_index

    with pytest.raises(ValueError, match="no edge index"):
        dataclasses.replace(tensors, pyg_graph=graph)


def test_tensors_that_disagree_on_the_number_of_users_raise():
    tensors = build_survey_tensors("cmap")
    one_user_short = tensors.user_attributes_cat[1:]

    with pytest.raises(ValueError, match=f"User count mismatch: .*user_attributes_cat {NUM_USERS - 1}"):
        dataclasses.replace(tensors, user_attributes_cat=one_user_short)


def test_reading_an_uncached_dataset_without_a_builder_raises(tmp_path):
    with pytest.raises(ValueError, match="no `build` was given"):
        VisitDataset(str(tmp_path))


def test_prepare_splits_loads_the_dataset_only_while_a_cache_is_stale(tmp_path, compose_config, monkeypatch):
    cfg = compose_config("data=cmap")
    build_dataset("cmap", tmp_path / cfg.data.paths.tensors)
    loads = []

    def counting_load_dataset(*args, **kwargs):
        loads.append(args)
        return load_dataset(*args, **kwargs)

    monkeypatch.setattr(datamodule, "load_dataset", counting_load_dataset)
    split = HouseholdSplit(val_size=0.2, test_size=0.2)

    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)
    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    assert len(loads) == 1


def test_prepare_splits_rebuilds_a_cache_missing_a_tensor_file(tmp_path, compose_config, monkeypatch):
    cfg = compose_config("data=cmap")
    root = tmp_path / cfg.data.paths.tensors
    dataset = build_dataset("cmap", root)
    monkeypatch.setattr(datamodule, "load_dataset", lambda *args, **kwargs: build_dataset("cmap", root))
    split = HouseholdSplit(val_size=0.2, test_size=0.2)
    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    missing = Path(dataset.processed_dir) / ProcessedFile.USER_ATTRIBUTES_CAT
    missing.unlink()
    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    assert missing.exists()


def test_snapshot_covers_every_flag_combination(snapshot):
    for inputs in snapshot.values():
        assert sum(key not in ("edge_attr", "y") for key in inputs) == 2 ** len(FLAGS)


if __name__ == "__main__":
    import tempfile

    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshots = {name: build_model_inputs(name, Path(tmp) / name) for name in SURVEYS}
    torch.save(snapshots, SNAPSHOT_PATH)
    print(f"Wrote {SNAPSHOT_PATH}")
