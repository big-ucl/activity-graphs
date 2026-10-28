"""Snapshot of the model inputs, end to end from a synthetic survey to ``extract_features``.

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
from activitygraphs.datasets import HouseholdSplit, datamodule
from activitygraphs.datasets.datamodule import positional_encodings
from activitygraphs.datasets.dataset import ProcessedFile, VisitDataset, load_dataset
from activitygraphs.datasets.scalers import apply_scalers, fit_scalers
from activitygraphs.ml.lightning_module import extract_features
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.surveys.cmap import CMAP_DEMOGRAPHICS_SCHEMA, CMAPSurvey
from activitygraphs.tensors import SurveyTensors, build_tensors
from activitygraphs.tensors.build import NO_DEMOGRAPHICS_SCHEMA
from activitygraphs.utils import add_lon_lat_from_centroid

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "model_inputs.pt"

GRID_SIDE = 3
NUM_HOUSEHOLDS = 8
USERS_PER_HOUSEHOLD = 2
NUM_USERS = NUM_HOUSEHOLDS * USERS_PER_HOUSEHOLD
TRAIN_IDX = list(range(10))
VISIT_PURPOSES = [Purpose.WORK_MAIN, Purpose.SHOP, Purpose.LEISURE_OTHER, Purpose.STUDY]
FLAGS = ("full_info", "use_demographics", "use_home_pe", "use_rwpe")
SURVEYS = ("cmap", "no_demographics")


class NoDemographicsSurvey(CMAPSurvey):
    """A ``CMAPSurvey`` without demographic columns, as TPG has none."""

    @property
    def demographics_schema(self) -> TensorSchema:
        """An empty schema."""
        return TensorSchema("user_attributes", ())


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
    hh_income = rng.choice([12.5, 37.5, 62.5, 87.5, 125.0, 175.0], NUM_HOUSEHOLDS)

    users = pl.DataFrame({
        "user_id": [f"u{k:02d}" for k in range(NUM_USERS)],
        "hh_id": [f"h{h:02d}" for h in hh],
        "home_loc_id": hh_home[hh],
        "has_missing_demographics": rng.random(NUM_USERS) < 0.2,
        "hh_num_adults": hh_adults[hh],
        "hh_num_children": hh_children[hh],
        "hh_num_vehicles": hh_vehicles[hh],
        "hh_num_bicycle": hh_bicycle[hh],
        "hh_income_kusd": hh_income[hh],
        "age": rng.integers(18, 80, NUM_USERS),
        "is_employed": rng.random(NUM_USERS) < 0.6,
        "is_student": rng.random(NUM_USERS) < 0.2,
        "can_drive": rng.random(NUM_USERS) < 0.7,
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
    survey_cls = CMAPSurvey if survey_name == "cmap" else NoDemographicsSurvey
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
        The scaled edge attributes, the labels, and the ``extract_features`` output over all users for every
        combination of ``FLAGS``, keyed by the active flags joined with ``+`` (``"none"`` for no flags).
    """
    dataset = build_dataset(survey_name, root)
    is_home_idx = dataset.schema.user_nodes.index_of("is_home")
    scalers = fit_scalers(dataset, TRAIN_IDX, fingerprint={}, exclude_user_node_cols=[is_home_idx])
    apply_scalers(dataset, scalers)

    batch = next(iter(pyg.loader.DataLoader(dataset, batch_size=len(dataset))))
    home_hop_distance = torch.as_tensor(dataset.home_hop_distance, dtype=torch.float)

    inputs = {"edge_attr": batch.edge_attr, "y": batch.y}
    for active in itertools.product([False, True], repeat=len(FLAGS)):
        flags = dict(zip(FLAGS, active, strict=True))
        key = "+".join(name for name, on in flags.items() if on) or "none"
        inputs[key] = extract_features(
            batch,
            full_info=flags["full_info"],
            is_home_idx=is_home_idx,
            use_demographics=flags["use_demographics"],
            home_hop_distance=home_hop_distance if flags["use_home_pe"] else None,
            use_rwpe=flags["use_rwpe"],
        )

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
    assert dataset.num_features == dataset.schema.locations.width + dataset.schema.user_nodes.width


def test_an_item_carries_the_distances_in_metres_beside_the_scaled_ones(tmp_path):
    dataset = build_dataset("cmap", tmp_path)
    metres = dataset.distances.clone()
    is_home_idx = dataset.schema.user_nodes.index_of("is_home")
    scalers = fit_scalers(dataset, TRAIN_IDX, fingerprint={}, exclude_user_node_cols=[is_home_idx])
    apply_scalers(dataset, scalers)
    item = dataset[3]

    scaled = torch.from_numpy(scalers.distances.transform(metres[3].numpy())).float()
    torch.testing.assert_close(item.distances_m, metres[3])
    torch.testing.assert_close(item.distances, scaled)


@pytest.mark.parametrize(
    ("survey_name", "user_attributes"),
    [("cmap", CMAP_DEMOGRAPHICS_SCHEMA), ("no_demographics", NO_DEMOGRAPHICS_SCHEMA)],
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
    assert schema.user_attributes.width == dataset.user_attributes.shape[-1]
    assert schema.distances.width == dataset.distances.shape[-1]
    assert schema.targets.width == dataset.visit_labels.shape[-1]


def test_a_tensor_whose_width_differs_from_its_schema_raises(tmp_path):
    tensors = build_survey_tensors("cmap")
    extra_column = ColumnSpec("extra", ColumnKind.BINARY)
    user_nodes = TensorSchema("user_nodes", (*tensors.schema.user_nodes.columns, extra_column))
    tensors = dataclasses.replace(tensors, schema=dataclasses.replace(tensors.schema, user_nodes=user_nodes))

    with pytest.raises(ValueError, match="`user_nodes` tensor has 1 columns but its schema has 2"):
        VisitDataset(str(tmp_path), lambda: tensors)


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
