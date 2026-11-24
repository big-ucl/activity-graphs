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
import pandas as pd
import polars as pl
import pytest
import torch
import torch_geometric as pyg
from geopandas.testing import assert_geodataframe_equal
from graphs import join_users
from shapely.geometry import box

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, USER_JOURNEY_SCHEMA, USER_SCHEMA, Mode, Purpose
from activitygraphs.config import HomePEConfig
from activitygraphs.datasets import GraphBatch, HouseholdSplit, Locations, UserDataset, Users, datamodule
from activitygraphs.datasets.transforms import FeatureTransforms, fit_transforms
from activitygraphs.graph import GraphTables
from activitygraphs.graph.rail import EDGE_INDEX_NAMES
from activitygraphs.graph.travel_times import TRAVEL_TIME_SCHEMA
from activitygraphs.models.features import InputFeatures, build_edge_encoder, build_feature_encoder
from activitygraphs.models.schema_encoder import _default_emb_dim
from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, EdgeKind, TensorSchema
from activitygraphs.surveys.cmap import CMAPSurvey
from activitygraphs.surveys.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA,
    EMPLOYMENT_STATUSES,
    INCOME_QUARTILES,
    STUDENT_STATUSES,
    VEHICLE_BANDS,
)
from activitygraphs.tensors import (
    EdgeTensors,
    SurveyTensors,
    build_tensors,
    check_tensors,
    load_tensors,
    read_tensors,
    save_tensors,
    tensors_path,
)
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
FLAGS = ("use_dist", "use_demographics", "use_home_pe", "use_rwpe", "use_access", "use_network_hops", "use_tt")
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


def build_graph(zones: gpd.GeoDataFrame) -> GraphTables:
    """Return the contiguity graph over ``zones`` with density, land-use and access columns, as ``graph/build.py`` does.

    The rail edges are one line, from ``build_rail_edges``.
    """
    rng = np.random.default_rng(0)
    projected = zones.to_crs(zones.estimate_utm_crs())
    projected["pop_density"] = rng.lognormal(-6.0, 1.0, len(zones))
    projected["poi_shop"] = rng.lognormal(-9.0, 1.5, len(zones))
    projected["land_use_residential"] = rng.uniform(0.0, 1.0, len(zones))
    projected["has_rail_station"] = rng.integers(0, 2, len(zones))
    projected["rail_station_distance_m"] = rng.uniform(100.0, 5000.0, len(zones))
    projected["peak_departures_per_hour"] = rng.poisson(20.0, len(zones)).astype(float)
    projected["motorway_junction_distance_m"] = rng.uniform(100.0, 10000.0, len(zones))
    opportunity_rng = np.random.default_rng(3)
    projected["jobs"] = opportunity_rng.lognormal(-7.0, 1.5, len(zones))
    projected["population"] = opportunity_rng.lognormal(-6.0, 1.0, len(zones))
    projected["area"] = projected.geometry.area
    nodes_gdf, contiguity_edges_gdf = c2g.contiguity_graph(projected.set_index("loc_id"), set_point_nodes=True)

    nodes = nodes_gdf.to_crs(CRS)
    # city2graph's edge order depends on the hash seed.
    contiguity_edges = contiguity_edges_gdf.sort_index().to_crs(CRS).rename_axis(EDGE_INDEX_NAMES)
    loc_ids = zones["loc_id"].tolist()
    rail_edges = build_rail_edges(loc_ids)
    travel_times = build_travel_times(loc_ids)

    edges = {EdgeKind.CONTIGUITY: contiguity_edges, EdgeKind.RAIL: rail_edges}

    return GraphTables(nodes=nodes, edges=edges, travel_times=travel_times)


def build_rail_edges(loc_ids: list[str]) -> pd.DataFrame:
    """Return the rail edges of one line calling at the first, middle and last zone, as ``build_rail_edges`` does."""
    line = [loc_ids[0], loc_ids[len(loc_ids) // 2], loc_ids[-1]]
    pairs = list(itertools.combinations(line, 2))

    pair_index = pd.MultiIndex.from_tuples(pairs, names=EDGE_INDEX_NAMES)

    return pd.DataFrame({"rail_time_s": [120.0, 300.0, 180.0], "rail_trips": [24, 24, 12]}, index=pair_index)


def build_travel_times(loc_ids: list[str]) -> pd.DataFrame:
    """Return a travel time of every mode for every ordered pair of zones, as ``graph/build.py`` does."""
    pairs = list(itertools.product(loc_ids, repeat=2))
    pair_index = pd.MultiIndex.from_tuples(pairs, names=EDGE_INDEX_NAMES)
    mode_columns = {}
    for mode, slowest_min in [("pt", 180.0), ("car", 60.0), ("walk", 240.0), ("bike", 120.0)]:
        time_min = np.linspace(0.0, slowest_min, len(pairs))
        mode_columns[f"{mode}_time_min"] = np.minimum(time_min, 180.0)
        mode_columns[f"{mode}_reachable"] = time_min < 180.0

    return pd.DataFrame(mode_columns, index=pair_index)


def build_users(loc_ids: list[str]) -> pl.DataFrame:
    """Return ``NUM_USERS`` users in households of ``USERS_PER_HOUSEHOLD`` with CMAP demographics."""
    rng = np.random.default_rng(1)
    hh = np.repeat(np.arange(NUM_HOUSEHOLDS), USERS_PER_HOUSEHOLD)
    hh_home = rng.choice(loc_ids, NUM_HOUSEHOLDS)
    hh_adults = rng.integers(1, 4, NUM_HOUSEHOLDS)
    hh_children = rng.integers(0, 3, NUM_HOUSEHOLDS)
    hh_vehicles = rng.choice(VEHICLE_BANDS, NUM_HOUSEHOLDS)
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


def build_survey_tensors(survey_name: str, excluded_location_features: tuple[str, ...] = ()) -> SurveyTensors:
    """Return the tensors of the synthetic survey ``survey_name``, without the ``excluded_location_features``."""
    zones = build_zones()
    graph = build_graph(zones)
    loc_ids = zones["loc_id"].tolist()
    users = build_users(loc_ids)
    survey_cls = SURVEY_CLASSES[survey_name]
    survey = survey_cls(zones, build_journeys(users, loc_ids), users).with_filter("zone")

    return build_tensors(survey, graph, excluded_location_features)


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

    edge_attr = locations.edges[EdgeKind.CONTIGUITY].attr.repeat(batch.num_users, 1)
    y = batch.visited.reshape(-1, 1)
    network_edges = build_edge_encoder(InputFeatures(), schema)(locations)
    inputs = {
        "edge_attr": edge_attr,
        "network_edge_index": network_edges.index,
        "network_edge_attr": network_edges.attr,
        "y": y,
    }
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


def test_the_graph_tables_read_back_as_saved(tmp_path):
    graph = build_graph(build_zones())

    graph.save(tmp_path)
    read_graph = GraphTables.read(tmp_path)

    assert_geodataframe_equal(read_graph.nodes, graph.nodes)
    assert_geodataframe_equal(read_graph.edges[EdgeKind.CONTIGUITY], graph.edges[EdgeKind.CONTIGUITY])
    pd.testing.assert_frame_equal(read_graph.edges[EdgeKind.RAIL], graph.edges[EdgeKind.RAIL])
    pd.testing.assert_frame_equal(read_graph.travel_times, graph.travel_times)


def test_the_graph_tables_write_the_files_they_list(tmp_path):
    build_graph(build_zones()).save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(GraphTables.file_names())


def test_the_tensor_bundles_write_the_files_they_list(tmp_path):
    locations, users, _ = build_survey_tensors("cmap")

    locations.save(tmp_path / "locations")
    users.save(tmp_path / "users")

    assert sorted(path.name for path in (tmp_path / "locations").iterdir()) == sorted(locations.file_names())
    assert sorted(path.name for path in (tmp_path / "users").iterdir()) == sorted(users.file_names())


def test_the_shared_graph_is_undirected(tmp_path):
    locations, _, _ = build_survey("cmap", tmp_path)
    graph = build_graph(build_zones())

    contiguity_edges = locations.edges[EdgeKind.CONTIGUITY]
    assert pyg.utils.is_undirected(contiguity_edges.index, num_nodes=locations.num_nodes)
    assert contiguity_edges.num_edges == 2 * len(graph.edges[EdgeKind.CONTIGUITY])
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
    features = InputFeatures(use_home_pe=False, use_dist=False, use_demographics=False, use_rwpe=False, use_tt=False)
    encoder = build_feature_encoder(features, locations, schema, HOME_PE)
    contiguity_edges = locations.edges[EdgeKind.CONTIGUITY]
    graphs = GraphBatch.expand(encoder(locations, batch), contiguity_edges)
    num_nodes = locations.num_nodes
    third_copy = slice(3 * num_nodes, 4 * num_nodes)

    is_home = torch.zeros(num_nodes, 1)
    is_home[users.home_node[3]] = 1.0

    expected_x = torch.cat([locations.x, locations.access, is_home], dim=-1)
    torch.testing.assert_close(graphs.x[third_copy], expected_x)
    assert graphs.x.shape[0] == NUM_USERS * num_nodes


def test_a_joined_batch_carries_the_distances_in_metres_beside_the_scaled_ones(tmp_path):
    locations, users, _, transforms = build_scaled_survey("cmap", tmp_path)
    batch = join_users(locations, users)
    metres = locations.distance_m[users.home_node[3]]

    distances = metres.unsqueeze(-1)
    scaled = transforms.groups["distances"].apply(distances).squeeze(-1)
    torch.testing.assert_close(batch.home_distance_m[3], metres)
    torch.testing.assert_close(batch.home_distance[3], scaled)


def test_the_travel_times_hold_the_row_of_each_ordered_pair_from_its_source(tmp_path):
    locations, _, _ = build_survey("cmap", tmp_path)
    loc_ids = build_zones()["loc_id"].tolist()
    travel_times = build_travel_times(loc_ids)
    source, target = loc_ids[2], loc_ids[7]
    expected_travel_times = torch.tensor(
        travel_times.loc[(source, target), TRAVEL_TIME_SCHEMA.names].to_numpy(dtype=np.float32)
    )

    torch.testing.assert_close(locations.travel_times_min[2, 7], expected_travel_times, rtol=0, atol=0)


def test_a_joined_batch_carries_the_travel_times_from_home_raw_and_scaled(tmp_path):
    locations, users, _, transforms = build_scaled_survey("cmap", tmp_path)
    batch = join_users(locations, users)
    travel_times_min = locations.travel_times_min[users.home_node[3]]

    scaled_travel_times = transforms.groups["travel_times"].apply(travel_times_min)
    reachable_idx = TRAVEL_TIME_SCHEMA.indices(ColumnKind.BINARY)
    torch.testing.assert_close(batch.home_travel_times_min[3], travel_times_min)
    torch.testing.assert_close(batch.home_travel_times[3], scaled_travel_times)
    torch.testing.assert_close(
        scaled_travel_times[:, reachable_idx], travel_times_min[:, reachable_idx], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("survey_name", "user_attributes"),
    [("cmap", DEMOGRAPHICS_SCHEMA), ("no_demographics", TensorSchema("user_attributes", ()))],
)
def test_the_cache_carries_the_schema_of_every_tensor(survey_name, user_attributes, tmp_path):
    locations, users, schema = build_survey(survey_name, tmp_path)

    assert schema.user_attributes == user_attributes
    assert schema.locations.names == ["pop_density", "poi_shop", "land_use_residential", "jobs", "population", "area"]
    assert schema.locations.indices(ColumnKind.CONTINUOUS) == [0, 1, 2, 3, 4, 5]
    assert schema.locations.width == locations.x.shape[1]
    for kind, edges in locations.edges.items():
        assert schema.edges[kind].width == edges.attr.shape[1]
    for use_contiguity_edges, use_rail_edges in itertools.product([False, True], repeat=2):
        features = InputFeatures(use_contiguity_edges=use_contiguity_edges, use_rail_edges=use_rail_edges)
        edge_encoder = build_edge_encoder(features, schema)
        assert edge_encoder(locations).attr.shape[1] == edge_encoder.out_dim
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
        embedded_width += _default_emb_dim(spec.cardinality)

    assert None in employment
    assert categorical.names == [
        "age_band",
        "employment_status",
        "student_status",
        "can_drive",
        "has_transit_pass",
        "hh_num_vehicles",
        "income_quartile",
    ]
    assert users.attributes.shape[-1] == DEMOGRAPHICS_SCHEMA.numeric.width
    torch.testing.assert_close(users.attributes_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(batch.attributes_cat[:, employment_idx], expected_codes, rtol=0, atol=0)
    assert encoder.out_dim == plain.out_dim + DEMOGRAPHICS_SCHEMA.numeric.width + embedded_width
    assert encoder(locations, batch).shape == (batch.num_users, batch.num_nodes, encoder.out_dim)


def test_an_excluded_location_feature_is_left_out_of_the_tensors():
    locations, _, schema = build_survey_tensors("cmap", excluded_location_features=("poi_shop",))

    assert schema.locations.names == ["pop_density", "land_use_residential", "jobs", "population", "area"]
    assert locations.x.shape[1] == 5


def test_excluding_a_column_the_graph_nodes_lack_raises():
    with pytest.raises(KeyError, match="poi_food"):
        build_survey_tensors("cmap", excluded_location_features=("poi_food",))


def test_a_tensor_whose_width_differs_from_its_schema_raises():
    locations, users, schema = build_survey_tensors("cmap")
    extra_column = ColumnSpec("extra", ColumnKind.BINARY)
    distances = TensorSchema("distances", (*schema.distances.columns, extra_column))
    wider_schema = dataclasses.replace(schema, distances=distances)

    with pytest.raises(ValueError, match="`distances`"):
        check_tensors(locations, users, wider_schema)


def test_a_home_node_outside_the_graph_raises():
    locations, users, schema = build_survey_tensors("cmap")
    home_node = users.home_node.clone()
    home_node[0] = locations.num_nodes

    with pytest.raises(ValueError, match="home node"):
        check_tensors(locations, dataclasses.replace(users, home_node=home_node), schema)


def test_an_edge_kind_that_repeats_a_pair_of_nodes_raises():
    locations, users, schema = build_survey_tensors("cmap")
    rail_edges = locations.edges[EdgeKind.RAIL]
    repeated_index = torch.cat([rail_edges.index, rail_edges.index[:, :1]], dim=1)
    repeated_attr = torch.cat([rail_edges.attr, rail_edges.attr[:1]])
    edges = {**locations.edges, EdgeKind.RAIL: EdgeTensors(repeated_index, repeated_attr)}

    with pytest.raises(ValueError, match="`rail` edges hold 1 repeated"):
        check_tensors(dataclasses.replace(locations, edges=edges), users, schema)


def test_a_nan_user_attribute_raises():
    locations, users, schema = build_survey_tensors("cmap")
    attributes = users.attributes.clone()
    attributes[0, 0] = float("nan")

    with pytest.raises(ValueError, match="`attributes`"):
        check_tensors(locations, dataclasses.replace(users, attributes=attributes), schema)


def test_tensors_that_disagree_on_the_number_of_users_raise():
    locations, users, schema = build_survey_tensors("cmap")
    one_user_short = users.attributes_cat[1:]

    with pytest.raises(ValueError, match=f"attributes_cat {NUM_USERS - 1}"):
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
    with pytest.raises(FileNotFoundError):
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

    datamodule._prepare_splits(cfg, split, seed=42, project_root=tmp_path)
    datamodule._prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    assert len(loads) == 1


def test_prepare_splits_rebuilds_a_cache_missing_a_tensor_file(tmp_path, compose_config, monkeypatch):
    cfg = compose_config("data=cmap")
    root = tensors_path(cfg.data, tmp_path)
    cache_survey_tensors("cmap", root)
    monkeypatch.setattr(datamodule, "load_tensors", lambda *args, **kwargs: cache_survey_tensors("cmap", root))
    split = HouseholdSplit(val_size=0.2, test_size=0.2)
    datamodule._prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    missing = root / USERS_DIR / UserFile.ATTRIBUTES_CAT
    missing.unlink()
    datamodule._prepare_splits(cfg, split, seed=42, project_root=tmp_path)

    assert missing.exists()


def test_snapshot_covers_every_flag_combination(snapshot):
    for inputs in snapshot.values():
        non_flag_keys = ("edge_attr", "network_edge_index", "network_edge_attr", "y")
        assert sum(key not in non_flag_keys for key in inputs) == 2 ** len(FLAGS)


if __name__ == "__main__":
    import tempfile

    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshots = {name: build_model_inputs(name, Path(tmp) / name) for name in SURVEYS}
    torch.save(snapshots, SNAPSHOT_PATH)
    print(f"Wrote {SNAPSHOT_PATH}")
