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
from graphs import join_users
from shapely.geometry import box

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, USER_JOURNEY_SCHEMA, USER_SCHEMA, Mode, Purpose
from activitygraphs.config import HomePEConfig
from activitygraphs.datasets import GraphBatch, HouseholdSplit, Locations, UserDataset, Users, datamodule
from activitygraphs.datasets.transforms import FeatureTransforms, fit_transforms
from activitygraphs.models.features import InputFeatures, build_feature_encoder
from activitygraphs.models.schema_encoder import default_emb_dim
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, TensorSchema
from activitygraphs.surveys.cmap import CMAPSurvey
from activitygraphs.surveys.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA,
    EMPLOYMENT_STATUSES,
    INCOME_QUARTILES,
    STUDENT_STATUSES,
)
from activitygraphs.tensors import (
    SurveyTensors,
    build_tensors,
    check_tensors,
    load_tensors,
    read_tensors,
    save_tensors,
    tensors_path,
)
from activitygraphs.tensors.build import build_location_tensors
from activitygraphs.tensors.bundles import UserFile
from activitygraphs.tensors.cache import FINGERPRINT_FILE, USERS_DIR
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
    hh_income = rng.choice(INCOME_QUARTILES, NUM_HOUSEHOLDS)
    is_missing = rng.random(NUM_USERS) < 0.2
    employment = rng.choice(EMPLOYMENT_STATUSES, NUM_USERS)

    users = pl.DataFrame({
        "user_id": [f"u{k:02d}" for k in range(NUM_USERS)],
        "hh_id": [f"h{h:02d}" for h in hh],
        "home_loc_id": hh_home[hh],
        "hh_num_adults": hh_adults[hh],
        "hh_num_children": hh_children[hh],
        "hh_num_vehicles": hh_vehicles[hh],
        "income_quartile": hh_income[hh],
        "age_band": rng.choice(AGE_BANDS, NUM_USERS),
        "employment_status": [
            None if missing else str(status) for missing, status in zip(is_missing, employment, strict=True)
        ],
        "student_status": rng.choice(STUDENT_STATUSES, NUM_USERS),
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


def cache_survey_tensors(survey_name: str, root: Path) -> SurveyTensors:
    """Return the tensors of the synthetic survey ``survey_name``, cached under ``root``."""
    tensors = build_survey_tensors(survey_name)
    save_tensors(root, *tensors)

    return tensors


def build_survey(survey_name: str, root: Path) -> tuple[Locations, Users, DatasetSchema]:
    """Return the unscaled locations, users and schema of the synthetic survey ``survey_name``, cached at ``root``."""
    locations, users, schema = cache_survey_tensors(survey_name, root)

    return Locations.from_tensors(locations, None), Users.from_tensors(users, None), schema


def build_scaled_survey(survey_name: str, root: Path) -> tuple[Locations, Users, DatasetSchema, FeatureTransforms]:
    """Return the synthetic survey ``survey_name`` scaled by transforms fitted on ``TRAIN_IDX``, cached under ``root``.

    Returns:
        Tuple ``(locations, users, schema, transforms)``.
    """
    location_tensors, user_tensors, schema = cache_survey_tensors(survey_name, root)
    transforms = fit_transforms(location_tensors, user_tensors, schema, TRAIN_IDX, fingerprint={})
    locations = Locations.from_tensors(location_tensors, transforms)

    return locations, Users.from_tensors(user_tensors, transforms), schema, transforms


def build_model_inputs(survey_name: str, root: Path) -> dict[str, torch.Tensor]:
    """Run the synthetic survey through the dataset, scaling and feature assembly.

    Args:
        survey_name: One of ``SURVEYS``.
        root: Directory for the processed tensors.

    Returns:
        The scaled edge attributes, the labels, and the ``FeatureEncoder`` output over all users for every
        combination of ``FLAGS``, keyed by the active flags joined with ``+`` (``"none"`` for no flags).
    """
    locations, users, schema, _ = build_scaled_survey(survey_name, root)

    batch = join_users(locations, users)

    edge_attr = locations.edge_attr.repeat(batch.num_users, 1)
    y = batch.visited.reshape(-1, 1)
    inputs = {"edge_attr": edge_attr, "y": y}
    for active in itertools.product([False, True], repeat=len(FLAGS)):
        flags = dict(zip(FLAGS, active, strict=True))
        key = "+".join(name for name, on in flags.items() if on) or "none"
        torch.manual_seed(0)
        encoder = build_feature_encoder(InputFeatures(**flags), locations, schema, HOME_PE)
        with torch.no_grad():
            x = encoder(locations, batch)
        inputs[key] = x.reshape(-1, x.shape[-1])
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
    locations, _, _ = build_survey("cmap", tmp_path)
    _, edges_gdf = build_graph(build_zones())

    assert pyg.utils.is_undirected(locations.edge_index, num_nodes=locations.num_nodes)
    assert locations.edge_index.size(1) == 2 * len(edges_gdf)
    assert locations.random_walk_pe.abs().sum() > 0


def test_the_cache_carries_the_survey_ids_and_the_fingerprint(tmp_path):
    locations, users, _ = build_survey("cmap", tmp_path)
    zones = build_zones()
    survey_users = build_users(zones["loc_id"].tolist())

    assert users.user_ids.equals(survey_users.sort("user_id").select("user_id", "hh_id"))
    assert locations.node_ids["loc_id"].to_list() == zones["loc_id"].tolist()

    fingerprint = json.loads((tmp_path / FINGERPRINT_FILE).read_text())
    assert fingerprint["n_users"] == NUM_USERS
    assert fingerprint["n_groups"] == NUM_HOUSEHOLDS


def test_an_item_carries_the_rows_of_one_user(tmp_path):
    _, users, _ = build_survey("cmap", tmp_path)
    item = UserDataset(users, [5, 3])[1]

    assert item.user_idx == 3
    assert item.home_node == users.home_node[3]
    torch.testing.assert_close(item.visited, users.visited[3])
    torch.testing.assert_close(item.attributes, users.attributes[3])
    torch.testing.assert_close(item.attributes_cat, users.attributes_cat[3])


def test_each_graph_copy_carries_the_location_and_user_node_features_apart(tmp_path):
    locations, users, schema = build_survey("cmap", tmp_path)
    batch = join_users(locations, users)
    features = InputFeatures(use_demographics=False, use_rwpe=False)
    encoder = build_feature_encoder(features, locations, schema, HOME_PE)
    graphs = GraphBatch.expand(encoder(locations, batch), locations)
    num_nodes = locations.num_nodes
    third_copy = slice(3 * num_nodes, 4 * num_nodes)

    is_home = torch.zeros(num_nodes, 1)
    is_home[users.home_node[3]] = 1.0

    torch.testing.assert_close(graphs.x[third_copy], torch.cat([locations.x, is_home], dim=-1))
    assert graphs.x.shape[0] == NUM_USERS * num_nodes


def test_a_joined_batch_carries_the_distances_in_metres_beside_the_scaled_ones(tmp_path):
    locations, users, _, transforms = build_scaled_survey("cmap", tmp_path)
    batch = join_users(locations, users)
    metres = locations.distance_m[users.home_node[3]]

    distances = metres.unsqueeze(-1)
    scaled = transforms.groups["distances"].apply(distances).squeeze(-1)
    torch.testing.assert_close(batch.home_distance_m[3], metres)
    torch.testing.assert_close(batch.home_distance[3], scaled)


@pytest.mark.parametrize(
    ("survey_name", "user_attributes"),
    [("cmap", DEMOGRAPHICS_SCHEMA), ("no_demographics", TensorSchema("user_attributes", ()))],
)
def test_the_cache_carries_the_schema_of_every_tensor(survey_name, user_attributes, tmp_path):
    locations, users, schema = build_survey(survey_name, tmp_path)

    assert schema.user_attributes == user_attributes
    assert schema.locations.names == ["pop_density", "poi_shop", "land_use_residential"]
    assert schema.locations.indices(ColumnKind.CONTINUOUS) == [0, 1, 2]
    assert schema.locations.width == locations.x.shape[1]
    assert schema.edges.width == locations.edge_attr.shape[1]
    assert schema.user_nodes.names == ["is_home"]
    assert schema.user_attributes.numeric.width == users.attributes.shape[-1]
    assert schema.user_attributes.categorical.width == users.attributes_cat.shape[-1]
    assert schema.distances.width == 1
    assert schema.targets.width == 1


def test_without_demographics_use_demographics_leaves_the_inputs_unchanged(tmp_path):
    inputs = build_model_inputs("no_demographics", tmp_path)

    for key, tensor in inputs.items():
        if "use_demographics" in key:
            without = key.replace("use_demographics", "").replace("++", "+").strip("+") or "none"
            torch.testing.assert_close(tensor, inputs[without], rtol=0, atol=0, msg=lambda m, key=key: f"{key}: {m}")


def test_categorical_attributes_are_cached_as_codes_and_embedded(tmp_path):
    locations, users, schema, _ = build_scaled_survey("cmap", tmp_path)
    survey_users = build_users(build_zones()["loc_id"].tolist()).sort("user_id")
    employment = survey_users["employment_status"].to_list()
    categorical = schema.user_attributes.categorical
    employment_idx = categorical.index_of("employment_status")
    categories = categorical.columns[employment_idx].categories
    expected_codes = torch.tensor([0 if status is None else categories.index(status) + 1 for status in employment])

    batch = join_users(locations, users)
    plain = build_feature_encoder(InputFeatures(use_demographics=False), locations, schema, HOME_PE)
    encoder = build_feature_encoder(InputFeatures(), locations, schema, HOME_PE)
    embedded_width = 0
    for spec in categorical.columns:
        embedded_width += default_emb_dim(spec.cardinality)

    assert None in employment
    assert categorical.names == [
        "age_band",
        "employment_status",
        "student_status",
        "can_drive",
        "has_transit_pass",
        "income_quartile",
    ]
    assert users.attributes.shape[-1] == DEMOGRAPHICS_SCHEMA.numeric.width
    torch.testing.assert_close(users.attributes_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(batch.attributes_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    assert encoder.out_dim == plain.out_dim + DEMOGRAPHICS_SCHEMA.numeric.width + embedded_width
    assert encoder(locations, batch).shape == (batch.num_users, batch.num_nodes, encoder.out_dim)


def test_a_tensor_whose_width_differs_from_its_schema_raises():
    locations, users, schema = build_survey_tensors("cmap")
    extra_column = ColumnSpec("extra", ColumnKind.BINARY)
    distances = TensorSchema("distances", (*schema.distances.columns, extra_column))
    wider_schema = dataclasses.replace(schema, distances=distances)

    with pytest.raises(ValueError, match=r"`distances` tensors have \(numeric, categorical\) widths \(1, 0\)"):
        check_tensors(locations, users, wider_schema)


def test_a_home_node_outside_the_graph_raises():
    locations, users, schema = build_survey_tensors("cmap")
    home_node = users.home_node.clone()
    home_node[0] = locations.num_nodes

    with pytest.raises(ValueError, match="1 users have a home node outside the graph"):
        check_tensors(locations, dataclasses.replace(users, home_node=home_node), schema)


def test_a_graph_without_an_edge_index_raises():
    locations, _, _ = build_survey_tensors("cmap")
    graph = pyg.data.Data(x=locations.x, edge_attr=locations.edge_attr)

    with pytest.raises(ValueError, match="no edge index"):
        build_location_tensors(graph, locations.distance_m, locations.node_ids)


def test_tensors_that_disagree_on_the_number_of_users_raise():
    locations, users, schema = build_survey_tensors("cmap")
    one_user_short = users.attributes_cat[1:]

    with pytest.raises(ValueError, match=f"User count mismatch: .*attributes_cat {NUM_USERS - 1}"):
        check_tensors(locations, dataclasses.replace(users, attributes_cat=one_user_short), schema)


def test_the_cached_tensors_read_back_unchanged(tmp_path):
    locations, users, schema = cache_survey_tensors("cmap", tmp_path)
    read_locations, read_users, read_schema = read_tensors(tmp_path)

    torch.testing.assert_close(read_locations.x, locations.x, rtol=0, atol=0)
    torch.testing.assert_close(read_locations.distance_m, locations.distance_m, rtol=0, atol=0)
    torch.testing.assert_close(read_users.visited, users.visited, rtol=0, atol=0)
    torch.testing.assert_close(read_users.home_node, users.home_node, rtol=0, atol=0)
    assert read_users.user_ids.equals(users.user_ids)
    assert read_schema == schema


def test_reading_an_uncached_tensor_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="is missing"):
        read_tensors(tmp_path)


def test_prepare_splits_loads_the_tensors_only_while_a_cache_is_stale(tmp_path, compose_config, monkeypatch):
    cfg = compose_config("data=cmap")
    cache_survey_tensors("cmap", tensors_path(cfg.data, tmp_path))
    loads = []

    def counting_load_tensors(*args, **kwargs):
        loads.append(args)
        return load_tensors(*args, **kwargs)

    monkeypatch.setattr(datamodule, "load_tensors", counting_load_tensors)
    split = HouseholdSplit(val_size=0.2, test_size=0.2)

    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)
    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    assert len(loads) == 1


def test_prepare_splits_rebuilds_a_cache_missing_a_tensor_file(tmp_path, compose_config, monkeypatch):
    cfg = compose_config("data=cmap")
    root = tensors_path(cfg.data, tmp_path)
    cache_survey_tensors("cmap", root)
    monkeypatch.setattr(datamodule, "load_tensors", lambda *args, **kwargs: cache_survey_tensors("cmap", root))
    split = HouseholdSplit(val_size=0.2, test_size=0.2)
    datamodule.prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    missing = root / USERS_DIR / UserFile.ATTRIBUTES_CAT
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
