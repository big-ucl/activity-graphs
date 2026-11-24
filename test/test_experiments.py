"""Tests for the variants, the seed loop and the aggregate / per-user result split."""

import contextlib
import dataclasses
import inspect
import types
from pathlib import Path
from typing import cast

import lightning as L
import polars as pl
import pytest
import torch
from encoders import continuous_schema
from graphs import chain_edge_index, join_users, make_users
from graphs import make_locations as make_graph_locations
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import Logger
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from torch.utils.data import DataLoader

from activitygraphs.base import FITTED_PARAM_STAGE, HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config, HomePEConfig, TransferConfig, read_config
from activitygraphs.datasets import HouseholdSplit, Locations, Users
from activitygraphs.datasets.datamodule import _compute_home_coverage
from activitygraphs.datasets.dataset import UserDataset
from activitygraphs.experiments import runners, training
from activitygraphs.experiments.baselines import _closed_form_baselines
from activitygraphs.experiments.restore import (
    RestoredVariant,
    SourceRun,
    _check_same_schema,
    build_restored_variants,
    check_checkpoints,
    check_node_weights,
    check_same_split,
    read_source_run,
    restore_scorer,
)
from activitygraphs.experiments.results import (
    _drop_empty_columns,
    create_output_dir,
    run_config,
    save_config,
    save_results,
)
from activitygraphs.experiments.runners import _run_variants
from activitygraphs.experiments.setup import ExperimentSetup, _check_tested_across_datasets
from activitygraphs.experiments.training import (
    WandBParams,
    _build_callbacks,
    _fitted_param_frame,
    _trainer_arguments,
    evaluate_model,
    train_model,
)
from activitygraphs.experiments.variants import (
    FEATURE_SUFFIXES,
    Variant,
    build_variants,
    check_distinct_names,
    variant_hyperparams,
    with_overrides,
)
from activitygraphs.graph.travel_times import TRAVEL_TIME_SCHEMA
from activitygraphs.models.architectures import BUILDERS, _lookup_architecture, build_scorer, has_node_weights
from activitygraphs.models.baselines import HomeZoneMFBaseline
from activitygraphs.models.features import (
    AccessFeatures,
    InputFeatures,
    LocationFeatures,
    RandomWalkPE,
    build_feature_encoder,
    build_node_feature_encoder,
)
from activitygraphs.models.scorers import GraphScorer, NodeScorer
from activitygraphs.models.two_tower import TwoTowerScorer
from activitygraphs.runs import (
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    DISTANCES_FILE,
    FITTED_PARAMS_FILE,
    PER_USER_FILE,
    RESULTS_FILE,
    SCORES_FILE,
)
from activitygraphs.schema import DatasetSchema, EdgeKind
from activitygraphs.tensors.build import USER_NODE_SCHEMA
from activitygraphs.training.losses import _build_bce_loss, build_loss
from activitygraphs.utils import get_project_root

RUN = "TPG-core-20261026-153000"
EXPERIMENT_CONFIGS = sorted((get_project_root() / "activitygraphs/conf/experiment").glob("*.yaml"))
HOME_PE = HomePEConfig(bins=8, max_hops=40)
DEFAULT_FEATURES = InputFeatures()


@pytest.fixture(scope="module")
def cfg() -> Config:
    return read_config(get_project_root(), verbose=False)


def make_setup(cfg: Config, train_seeds: list[int], variants: list[Variant]) -> ExperimentSetup:
    """An ExperimentSetup with no datamodule; ``build_scorer`` is stubbed by the tests, so no data is needed."""
    return ExperimentSetup(
        cfg=cfg,
        run_dir=None,
        datamodule=types.SimpleNamespace(locations=None, schema=None),  # type: ignore[arg-type]
        loss=None,  # type: ignore[arg-type]
        models=dict(cfg.models),
        variants=variants,
        restored_variants=[],
        scaled_datamodules={},
        baseline_results=[],
        home_distances=pl.DataFrame(),
        train_seeds=train_seeds,
        debug=False,
        wandb_params=WandBParams(use_wandb=False),
    )


def make_schema(
    num_location_features: int = 5,
    num_attributes: int = 3,
    num_edge_features: int = 2,
    num_access_features: int = 4,
    num_travel_time_features: int = 2,
) -> DatasetSchema:
    """The schema of a survey with continuous location, access, edge and user attribute columns."""
    return DatasetSchema(
        locations=continuous_schema("locations", num_location_features),
        access=continuous_schema("access", num_access_features),
        edges={
            EdgeKind.CONTIGUITY: continuous_schema("contiguity_edges", num_edge_features),
            EdgeKind.RAIL: continuous_schema("rail_edges", 0),
        },
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=continuous_schema("user_attributes", num_attributes),
        distances=continuous_schema("distances", 1),
        travel_times=continuous_schema("travel_times", num_travel_time_features),
        targets=continuous_schema("targets", 1),
    )


def make_locations() -> types.SimpleNamespace:
    """Stand-in exposing only what the scorer builders read off ``Locations``."""
    return types.SimpleNamespace(num_rwpe_features=2, num_nodes=6)


def make_survey(rail_edge_index: torch.Tensor | None = None) -> tuple[Locations, Users]:
    """A chain of nodes and two users, with random columns of the widths given by ``make_schema()``.

    ``rail_edge_index`` adds rail edges, by default none.
    """
    generator = torch.Generator().manual_seed(0)
    schema, stand_in = make_schema(), make_locations()
    num_nodes, num_users = stand_in.num_nodes, 2
    edge_index = chain_edge_index(num_nodes)
    x = torch.rand(num_nodes, schema.locations.width, generator=generator)
    edge_attr = torch.rand(edge_index.shape[1], schema.edges[EdgeKind.CONTIGUITY].width, generator=generator)
    random_walk_pe = torch.rand(num_nodes, stand_in.num_rwpe_features, generator=generator)
    distance_m = torch.rand(num_nodes, num_nodes, generator=generator) * 1000.0
    access = torch.rand(num_nodes, schema.access.width, generator=generator)
    travel_times = torch.rand(num_nodes, num_nodes, schema.travel_times.width, generator=generator)
    locations = make_graph_locations(
        x,
        edge_index,
        edge_attr,
        random_walk_pe,
        distance_m,
        access,
        rail_edge_index=rail_edge_index,
        travel_times_min=travel_times,
    )

    visited = (torch.rand(num_users, num_nodes, generator=generator) < 0.5).float()
    home_node = torch.tensor([0, 3])
    attributes = torch.rand(num_users, schema.user_attributes.width, generator=generator)

    return locations, make_users(visited, home_node, attributes)


def record_training(monkeypatch, frame=None) -> list[dict]:
    """Replace scorer building, logging, training and testing with stubs; return one record per tested model.

    Each record holds the logger ``run_name`` and ``hyperparams``, the ``model`` trained (None when the scorer is only
    tested), and the ``name`` and ``scorer`` tested.
    """
    calls: list[dict] = []

    @contextlib.contextmanager
    def model_logger(run_name, hyperparams, wandb_params, log_dir):
        calls.append({"run_name": run_name, "hyperparams": hyperparams, "model": None})
        yield None

    def train_model(scorer, run_name, model, *args):
        calls[-1]["model"] = model
        return []

    def evaluate_model(scorer, name, *args):
        calls[-1] |= {"name": name, "scorer": scorer}
        return frame(name) if frame else pl.DataFrame({"name": [name], "stage": ["test"]})

    monkeypatch.setattr(runners, "model_logger", model_logger)
    monkeypatch.setattr(runners, "train_model", train_model)
    monkeypatch.setattr(runners, "evaluate_model", evaluate_model)
    monkeypatch.setattr(
        runners, "build_scorer", lambda config, features, locations, schema, home_pe_cfg: torch.nn.Linear(1, 1)
    )

    return calls


def make_score_frame(name: str, seed: int | None = 42) -> pl.DataFrame:
    """Two score-vector rows, as ``_score_vector_frame`` returns them."""
    return pl.DataFrame({
        "user_idx": [7, 9],
        "scores": [[0.1, 0.9, 0.4], [0.7, 0.2, 0.5]],
    }).with_columns(
        name=pl.lit(name),
        stage=pl.lit(SCORE_VECTOR_STAGE),
        epoch=pl.lit(None, dtype=pl.Int64),
        seed=pl.lit(seed, dtype=pl.Int64),
    )


def make_home_distance_frame() -> pl.DataFrame:
    """Two rows of per-node distances from home, as ``home_distance_frame`` returns them."""
    return pl.DataFrame({
        "user_idx": [7, 9],
        "distances": [[0.0, 900.0, 4000.0], [4000.0, 900.0, 0.0]],
    }).with_columns(stage=pl.lit(HOME_DISTANCE_STAGE), epoch=pl.lit(None, dtype=pl.Int64))


def make_model_frame(name: str, seed: int | None = 42) -> pl.DataFrame:
    """One fit row, one test row, and two per-user rows, as ``train_and_evaluate_model`` returns them."""
    aggregate = pl.DataFrame([
        {"name": name, "stage": "fit", "epoch": 0, "train_loss": 0.5, "val_r_precision": 0.4},
        {"name": name, "stage": "test", "epoch": None, "test_r_precision": 0.48},
    ])
    per_user = pl.DataFrame({
        "user_idx": [7, 9],
        "n_pos": [2, 3],
        "r_precision": [1.0, 0.5],
        "recall": [1.0, 0.33],
    }).with_columns(name=pl.lit(name), stage=pl.lit(PER_USER_STAGE), epoch=pl.lit(None, dtype=pl.Int64))

    combined = pl.concat([aggregate, per_user], how="diagonal")
    return combined.with_columns(seed=pl.lit(seed, dtype=pl.Int64))


class TestVariant:
    def test_the_models_as_configured_have_no_suffixes(self, cfg):
        names = [Variant(cfg.models[key]).name for key in ["mlp", "gat", "gps", "two_tower", "home_zone_mf"]]

        assert names == ["MLP", "GATv2-8", "GTransformer-2", "TwoTower-4", "HomeZoneMF"]

    @pytest.mark.parametrize(("flag", "suffix"), FEATURE_SUFFIXES.items())
    def test_a_flag_off_its_default_adds_its_suffix(self, cfg, flag, suffix):
        assert Variant(cfg.models.mlp, InputFeatures(**{flag: False})).name == f"MLP-{suffix}"

    def test_every_flag_has_a_suffix(self):
        assert set(FEATURE_SUFFIXES) == {flag.name for flag in dataclasses.fields(InputFeatures)}

    def test_the_flag_suffixes_follow_the_field_order(self, cfg):
        features = InputFeatures(use_home_pe=False, use_dist=False, use_demographics=False, use_rwpe=False)

        assert Variant(cfg.models.mlp, features).name == "MLP-nohpe-nodist-nodemo-norwpe"

    def test_departures_list_the_flags_off_their_default(self):
        assert InputFeatures(use_dist=False, use_rwpe=False).departures == ["use_dist", "use_rwpe"]
        assert InputFeatures().departures == []

    def test_the_option_suffixes_come_before_the_flag_suffixes(self, cfg):
        overrides = {"bias": "node", "pair_location_features": True, "node_embedding": True, "embedding_dim": 128}
        two_tower = with_overrides(cfg.models.two_tower, overrides)

        name = Variant(two_tower, InputFeatures(use_demographics=False)).name

        assert name == "TwoTower-4-nodebias-pairloc-nodeemb-d128-nodemo"

    @pytest.mark.parametrize(
        ("bias", "name"),
        [("location", "TwoTower-4"), ("embedding_mlp", "TwoTower-4-mlpbias"), ("feature_mlp", "TwoTower-4-featbias")],
    )
    def test_each_bias_has_its_suffix(self, cfg, bias, name):
        assert Variant(with_overrides(cfg.models.two_tower, {"bias": bias})).name == name

    def test_name_follows_an_overridden_depth(self, cfg):
        gat = with_overrides(cfg.models.gat, {"num_gcn_layers": 4})

        assert Variant(gat).name == "GATv2-4"
        assert cfg.models.gat.name == "GATv2-8"


def variant_names(cfg, *entries: dict) -> list[str]:
    """Names of the variants that ``entries``, written as in an experiment yaml, expand to over ``cfg.models``."""
    variants = build_variants(cast(list, list(entries)), dict(cfg.models))
    return [variant.name for variant in variants]


class TestBuildVariants:
    def test_an_entry_sets_the_model_and_the_features(self, cfg):
        entry = {"model": "two_tower", "set": {"bias": "node"}, "features": {"use_rwpe": False}}
        (variant,) = build_variants(cast(list, [entry]), dict(cfg.models))

        assert variant.model.bias == "node"
        assert variant.features == InputFeatures(use_rwpe=False)
        assert variant.name == "TwoTower-4-nodebias-norwpe"

    def test_a_sweep_gives_one_variant_per_value_on_top_of_the_set_fields(self, cfg):
        entry = {"model": "two_tower", "set": {"node_embedding": True}, "sweep": {"embedding_dim": [32, 128]}}

        assert variant_names(cfg, entry) == ["TwoTower-4-nodeemb-d32", "TwoTower-4-nodeemb-d128"]

    def test_the_variants_keep_the_order_of_the_entries(self, cfg):
        names = variant_names(cfg, {"model": "gat"}, {"model": "mlp", "features": {"use_home_pe": False}})

        assert names == ["GATv2-8", "MLP-nohpe"]

    def test_setting_a_field_to_its_configured_value_is_allowed(self, cfg):
        assert variant_names(cfg, {"model": "two_tower", "set": {"bias": "location"}}) == ["TwoTower-4"]

    def test_an_unknown_key_raises(self, cfg):
        with pytest.raises(ConfigKeyError):
            variant_names(cfg, {"model": "mlp", "sweeps": {}})

    def test_an_unknown_flag_raises(self, cfg):
        with pytest.raises(TypeError):
            variant_names(cfg, {"model": "mlp", "features": {"use_hpe": False}})

    def test_a_field_that_leaves_the_name_unchanged_raises(self, cfg):
        with pytest.raises(ValueError, match="hidden_channels"):
            variant_names(cfg, {"model": "mlp", "set": {"hidden_channels": 256}})

    def test_a_sweep_of_two_fields_raises(self, cfg):
        with pytest.raises(ValueError, match="one field"):
            variant_names(cfg, {"model": "two_tower", "sweep": {"embedding_dim": [32], "num_gcn_layers": [2]}})

    def test_two_variants_with_one_name_raise(self, cfg):
        with pytest.raises(ValueError, match="MLP"):
            variant_names(cfg, {"model": "mlp"}, {"model": "mlp", "features": {}})


def scorer_of(model, features: InputFeatures = DEFAULT_FEATURES, schema: DatasetSchema | None = None):
    """Build the scorer of ``model`` on the stand-in locations and ``schema``, by default ``make_schema()``."""
    locations = make_locations()
    schema = make_schema() if schema is None else schema
    return build_scorer(model, features, locations, schema, HOME_PE)  # type: ignore[arg-type]


class TestBuildScorer:
    @pytest.mark.parametrize("key", ["mlp", "gat", "gps", "home_zone_mf", "two_tower"])
    def test_every_architecture_builds(self, cfg, key):
        assert isinstance(scorer_of(cfg.models[key]), torch.nn.Module)

    @pytest.mark.parametrize("key", ["gat", "two_tower"])
    @pytest.mark.parametrize("use_contiguity_edges", [False, True])
    @pytest.mark.parametrize("use_rail_edges", [False, True])
    def test_one_layer_passes_messages_along_the_edges_of_each_kind_in_use(
        self, cfg, key, use_contiguity_edges, use_rail_edges
    ):
        rail_edge_index = torch.tensor([[0, 5], [5, 0]])
        locations, users = make_survey(rail_edge_index)
        model = with_overrides(cfg.models[key], {"num_gcn_layers": 1})
        features = InputFeatures(use_contiguity_edges=use_contiguity_edges, use_rail_edges=use_rail_edges)
        torch.manual_seed(0)
        scorer = build_scorer(model, features, locations, make_schema(), HOME_PE).eval()
        far_end_x = locations.x.clone()
        far_end_x[0] += 1.0
        moved_locations = dataclasses.replace(locations, x=far_end_x)

        with torch.no_grad():
            logits = scorer(locations, join_users(locations, users))
            moved_logits = scorer(moved_locations, join_users(moved_locations, users))

        away_user = 1
        rail_end_moved = not torch.allclose(moved_logits[away_user, 5], logits[away_user, 5])
        contiguous_node_moved = not torch.allclose(moved_logits[away_user, 1], logits[away_user, 1])
        assert users.home_node[away_user] == 3
        assert rail_end_moved == use_rail_edges
        assert contiguous_node_moved == use_contiguity_edges

    def test_input_width_follows_the_variant_features(self, cfg):
        locations, schema = make_locations(), make_schema()
        plain_features = InputFeatures(use_home_pe=False, use_dist=False, use_demographics=False, use_tt=False)
        plain = build_feature_encoder(plain_features, locations, schema, HOME_PE)  # type: ignore[arg-type]
        dist_features = InputFeatures(use_home_pe=False, use_tt=False)
        dist = build_feature_encoder(dist_features, locations, schema, HOME_PE)  # type: ignore[arg-type]
        node_width = schema.locations.width + schema.access.width + locations.num_rwpe_features

        assert plain.out_dim == node_width + schema.user_nodes.width
        assert dist.out_dim - plain.out_dim == schema.user_attributes.width + schema.distances.width

    def test_rwpe_free_input_width_leaves_out_the_encoding(self, cfg):
        locations, schema = make_locations(), make_schema()
        with_rwpe = build_feature_encoder(InputFeatures(use_home_pe=False), locations, schema, HOME_PE)  # type: ignore[arg-type]
        features = InputFeatures(use_home_pe=False, use_rwpe=False)
        without_rwpe = build_feature_encoder(features, locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert with_rwpe.out_dim - without_rwpe.out_dim == locations.num_rwpe_features

    def test_the_access_flag_adds_the_access_columns(self, cfg):
        locations, schema = make_locations(), make_schema()
        without_access = build_feature_encoder(InputFeatures(use_access=False), locations, schema, HOME_PE)  # type: ignore[arg-type]
        with_access = build_feature_encoder(InputFeatures(), locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert with_access.out_dim - without_access.out_dim == schema.access.width

    def test_the_travel_time_flag_adds_the_travel_time_columns(self, cfg):
        locations, schema = make_locations(), make_schema()
        without_tt = build_feature_encoder(InputFeatures(use_tt=False), locations, schema, HOME_PE)  # type: ignore[arg-type]
        with_tt = build_feature_encoder(InputFeatures(), locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert with_tt.out_dim - without_tt.out_dim == schema.travel_times.width

    @pytest.mark.parametrize(
        ("features", "source_types"),
        [
            (InputFeatures(), [LocationFeatures, AccessFeatures, RandomWalkPE]),
            (InputFeatures(use_access=False), [LocationFeatures, RandomWalkPE]),
            (InputFeatures(use_rwpe=False), [LocationFeatures, AccessFeatures]),
            (InputFeatures(use_access=False, use_rwpe=False), [LocationFeatures]),
        ],
    )
    def test_the_node_sources_follow_the_flags(self, features, source_types):
        encoder = build_node_feature_encoder(features, make_locations(), make_schema(), None)  # type: ignore[arg-type]

        assert [type(source) for source in encoder.sources] == source_types

    def test_home_pe_adds_its_bins(self, cfg):
        locations, schema = make_locations(), make_schema()
        plain = build_feature_encoder(InputFeatures(use_home_pe=False), locations, schema, HOME_PE)  # type: ignore[arg-type]
        with_pe = build_feature_encoder(InputFeatures(), locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert with_pe.out_dim - plain.out_dim == HOME_PE.bins

    @pytest.mark.parametrize("key", ["mlp", "gat", "gps"])
    def test_every_network_scores_every_node_of_every_user(self, cfg, key):
        locations, users = make_survey()
        scorer = build_scorer(cfg.models[key], DEFAULT_FEATURES, locations, make_schema(), HOME_PE)
        batch = join_users(locations, users)

        assert scorer(locations, batch).shape == (users.num_users, locations.num_nodes)

    def test_gat_norm_follows_the_config(self, cfg):
        normed = scorer_of(cfg.models.gat)
        plain = scorer_of(with_overrides(cfg.models.gat, {"norm": None}))

        assert any(isinstance(module, torch.nn.LayerNorm) for module in normed.modules())
        assert not any(isinstance(module, torch.nn.LayerNorm) for module in plain.modules())

    @pytest.mark.parametrize(
        ("key", "scorer_type"),
        [
            ("mlp", NodeScorer),
            ("gat", GraphScorer),
            ("gps", GraphScorer),
            ("home_zone_mf", HomeZoneMFBaseline),
            ("two_tower", TwoTowerScorer),
        ],
    )
    def test_each_architecture_runs_in_its_scorer(self, cfg, key, scorer_type):
        assert isinstance(scorer_of(cfg.models[key]), scorer_type)

    def test_unknown_gat_norm_raises(self, cfg):
        with pytest.raises(KeyError, match="normalisation"):
            scorer_of(with_overrides(cfg.models.gat, {"norm": "nope"}))

    def test_home_zone_mf_rejects_input_features(self, cfg):
        with pytest.raises(ValueError, match="HomeZoneMF"):
            scorer_of(cfg.models.home_zone_mf, InputFeatures(use_dist=False))


class TestTrainerConfig:
    def test_every_key_is_a_trainer_argument(self, cfg):
        assert set(cfg.trainer) <= set(inspect.signature(L.Trainer.__init__).parameters)

    @pytest.mark.parametrize(("data", "fraction"), [("tts2016", 1.0), ("cmap", 1.0)])
    def test_the_trainer_gets_the_batch_fraction_of_the_dataset(self, compose_config, data, fraction):
        cfg = compose_config(f"data={data}")

        assert _trainer_arguments(cfg)["limit_train_batches"] == fraction

    @pytest.mark.parametrize(("data", "max_epochs"), [("tts2022", 50), ("cmap", 50)])
    def test_the_trainer_gets_the_epochs_of_the_dataset(self, compose_config, data, max_epochs):
        cfg = compose_config(f"data={data}")

        assert _trainer_arguments(cfg)["max_epochs"] == max_epochs


EXPECTED_VARIANTS = {
    "baselines": ["HomeZoneMF"],
    "comparison": [
        "MLP-nohpe-nodist",
        "MLP-nohpe",
        "MLP",
        "GATv2-8-nohpe-nodist",
        "GATv2-8-nohpe",
        "GATv2-8-nodist",
        "GATv2-8",
        "GTransformer-2-nodist",
        "GTransformer-2",
        "HomeZoneMF",
    ],
    "access_ablation": ["MLP", "MLP-noacc", "GATv2-2", "GATv2-2-noacc", "TwoTower-4", "TwoTower-4-noacc"],
    "core": ["HomeZoneMF", "MLP-nohpe-norwpe", "MLP-nohpe", "MLP", "GATv2-8", "GTransformer-2"],
    "demographics_ablation": ["MLP", "MLP-nodemo", "GATv2-2", "GATv2-2-nodemo"],
    "depth_sweep": ["MLP", "GATv2-1", "GATv2-2", "GATv2-4", "GATv2-8"],
    "headline": ["MLP", "GATv2-2", "TwoTower-4", "HomeZoneMF"],
    "network_hops": ["MLP", "MLP-nonet", "MLP-nohpe"],
    "travel_times": ["MLP", "MLP-nott", "TwoTower-4", "TwoTower-4-nott"],
    "rail_redundancy": [
        "MLP-nonet-nott",
        "GATv2-2-nonet-nott",
        "GATv2-2-nonet-nott-norail",
        "GATv2-2-nonet-nott-nocontig-norail",
    ],
    "edge_ablation": [
        "MLP",
        "GATv2-2",
        "GATv2-2-norail",
        "GATv2-2-nocontig-norail",
        "TwoTower-4",
        "TwoTower-4-norail",
        "TwoTower-4-nocontig-norail",
    ],
    "overfit_health": ["MLP-nohpe", "GATv2-8-nohpe"],
    "two_tower_ablation": [
        "MLP",
        "GATv2-4",
        "TwoTower-4-nohpe-nodist-nodemo",
        "TwoTower-4-nodemo",
        "TwoTower-4-nohpe-nodist",
        "TwoTower-4",
        "TwoTower-4-nodebias",
        "TwoTower-4-nodeemb",
    ],
    "two_tower_bias_head": ["MLP", "TwoTower-4", "TwoTower-4-mlpbias", "TwoTower-4-featbias", "TwoTower-4-nodebias"],
    "two_tower_pair_inputs": ["MLP", "TwoTower-4", "TwoTower-4-pairloc", "TwoTower-4-d128"],
    "transfer_tts2022": [],
    "transfer_all_users": [],
    "transfer_finetune": ["MLP", "TwoTower-4"],
}


def experiment_variant_names(cfg: Config) -> list[str]:
    """Names of the variants of the experiment of ``cfg``, in training order."""
    return [variant.name for variant in build_variants(cfg.experiment.variants, dict(cfg.models))]


class TestExperimentConfigs:
    @pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda path: path.stem)
    def test_every_experiment_trains_its_variants(self, compose_config, path):
        cfg = compose_config(f"experiment={path.stem}")

        assert experiment_variant_names(cfg) == EXPECTED_VARIANTS[path.stem]

    def test_comparison_is_the_default(self, cfg):
        assert cfg.experiment.name == "comparison"

    def test_overfit_health_sets_what_the_check_reads(self, compose_config):
        cfg = compose_config("experiment=overfit_health")

        assert cfg.experiment.name == "overfit_health"
        assert cfg.trainer.overfit_batches > 0
        assert cfg.train.log_train_ranking is True
        assert len(cfg.train.train_seeds) == 1

    def test_the_command_line_overrides_a_sweep(self, compose_config):
        cfg = compose_config("experiment=depth_sweep", "experiment.variants.1.sweep.num_gcn_layers=[3]")

        assert experiment_variant_names(cfg) == ["MLP", "GATv2-3"]

    @pytest.mark.parametrize("experiment", ["headline", "demographics_ablation"])
    def test_the_gat_trains_at_depth_two(self, compose_config, experiment):
        cfg = compose_config(f"experiment={experiment}")

        assert cfg.models.gat.num_gcn_layers == 2
        assert cfg.models.gat.name == "GATv2-2"

    def test_the_command_line_overrides_the_gat_depth_of_an_experiment(self, compose_config):
        cfg = compose_config("experiment=headline", "models.gat.num_gcn_layers=8")

        assert cfg.models.gat.num_gcn_layers == 8

    def test_the_gat_of_the_two_tower_ablation_follows_the_node_tower(self, compose_config):
        cfg = compose_config("experiment=two_tower_ablation", "models.two_tower.num_gcn_layers=2")

        assert {"GATv2-2", "TwoTower-2"} <= set(experiment_variant_names(cfg))


class TestArchitectureRegistry:
    @pytest.mark.parametrize("path", sorted((get_project_root() / "activitygraphs/conf/model").glob("*.yaml")))
    def test_every_model_config_names_a_registered_architecture(self, path):
        assert OmegaConf.load(path).architecture in BUILDERS

    def test_an_unregistered_architecture_raises(self, cfg):
        model = with_overrides(cfg.models.mlp, {"architecture": "unknown"})

        with pytest.raises(KeyError):
            _lookup_architecture(model)


class TestRunVariants:
    def _variants(self, cfg) -> list[Variant]:
        return [Variant(cfg.models.mlp), Variant(cfg.models.gat, InputFeatures(use_home_pe=False))]

    def test_trains_every_variant_once_per_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        frames = _run_variants(make_setup(cfg, [1, 2], self._variants(cfg)))

        assert len(frames) == 4
        assert [call["name"] for call in calls] == ["MLP", "GATv2-8-nohpe", "MLP", "GATv2-8-nohpe"]

    def test_tags_each_frame_with_its_training_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        seeds = pl.concat(_run_variants(make_setup(cfg, [1, 2], self._variants(cfg))))["seed"].to_list()

        assert seeds == [1, 1, 2, 2]

    def test_builds_a_fresh_model_for_every_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        built: list[torch.nn.Module] = []

        def build(config, features, locations, schema, home_pe_cfg):
            built.append(torch.nn.Linear(1, 1))
            return built[-1]

        monkeypatch.setattr(runners, "build_scorer", build)
        _run_variants(make_setup(cfg, [1, 2], [Variant(cfg.models.mlp)]))

        assert len(built) == 2
        assert built[0] is not built[1]

    def test_every_variant_starts_from_the_seeded_random_state(self, cfg, monkeypatch):
        record_training(monkeypatch)
        draws: list[float] = []

        def build(config, features, locations, schema, home_pe_cfg):
            draws.append(torch.rand(1).item())
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(runners, "build_scorer", build)
        _run_variants(make_setup(cfg, [1], self._variants(cfg)))

        assert draws[0] == draws[1]

    def test_run_tag_separates_checkpoints_across_seeds(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [1, 2], self._variants(cfg)))

        assert [call["hyperparams"]["run_tag"] for call in calls] == ["seed1", "seed1", "seed2", "seed2"]
        assert [call["run_name"] for call in calls] == [
            "MLP-seed1",
            "GATv2-8-nohpe-seed1",
            "MLP-seed2",
            "GATv2-8-nohpe-seed2",
        ]

    def test_no_run_tag_for_a_single_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [42], self._variants(cfg)))

        assert [call["run_name"] for call in calls] == ["MLP", "GATv2-8-nohpe"]

    def test_forwards_each_variant(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [42], self._variants(cfg)))

        variants = self._variants(cfg)

        assert [call["name"] for call in calls] == [variant.name for variant in variants]
        assert [call["model"] for call in calls] == [variant.model for variant in variants]
        expected_hyperparams = []
        for variant in variants:
            expected_hyperparams.append({
                "model": variant.name,
                "run_tag": None,
                "model_type": "trained",
                **variant_hyperparams(variant),
            })
        assert [call["hyperparams"] for call in calls] == expected_hyperparams


class TestVariantHyperparams:
    def test_logs_the_architecture_and_flags(self, cfg):
        hyperparams = variant_hyperparams(Variant(cfg.models.gat, InputFeatures(use_dist=False)))

        assert hyperparams["architecture"]["num_gcn_layers"] == cfg.models.gat.num_gcn_layers
        assert hyperparams["use_dist"] is False


class TestDropEmptyColumns:
    def test_drops_all_null_columns_only(self):
        df = pl.DataFrame({"kept": [1, None], "dropped": [None, None]})

        assert _drop_empty_columns(df).columns == ["kept"]

    def test_keeps_an_all_null_seed(self):
        df = pl.DataFrame(
            {"seed": [None, None], "dropped": [None, None]}, schema={"seed": pl.Int64, "dropped": pl.Int64}
        )

        assert _drop_empty_columns(df).columns == ["seed"]

    def test_empty_frame_is_returned_unchanged(self):
        df = pl.DataFrame({"a": [], "b": []})

        assert _drop_empty_columns(df).columns == ["a", "b"]


class TestRunExperiment:
    def _run(self, compose_config, monkeypatch, *overrides: str) -> tuple[list[dict], list[bool]]:
        """Run ``run_experiment`` with stubs; return the training calls and whether each setup fitted the baselines."""
        cfg = compose_config(*overrides)
        calls = record_training(monkeypatch, frame=make_model_frame)
        baselines_fitted: list[bool] = []

        def setup_experiment(cfg, with_baselines: bool = True):
            baselines_fitted.append(with_baselines)
            variants = build_variants(cfg.experiment.variants, dict(cfg.models))
            return make_setup(cfg, [42, 43], variants)

        monkeypatch.setattr(runners, "setup_experiment", setup_experiment)
        monkeypatch.setattr(runners, "save_results", lambda *a, **kw: None)

        runners.run_experiment(cfg)

        return calls, baselines_fitted

    def test_trains_every_variant_once_per_seed_in_order(self, compose_config, monkeypatch):
        calls, _ = self._run(compose_config, monkeypatch, "experiment=headline")

        assert [call["name"] for call in calls] == ["MLP", "GATv2-2", "TwoTower-4", "HomeZoneMF"] * 2

    def test_fits_the_baselines(self, compose_config, monkeypatch):
        _, with_baselines = self._run(compose_config, monkeypatch, "experiment=core")

        assert with_baselines == [True]

    def test_an_overfitting_run_skips_the_baselines(self, compose_config, monkeypatch):
        _, with_baselines = self._run(compose_config, monkeypatch, "experiment=overfit_health")

        assert with_baselines == [False]

    def test_an_overfitting_run_needs_the_training_ranking(self, compose_config, monkeypatch):
        cfg = compose_config("experiment=overfit_health", "train.log_train_ranking=false")
        monkeypatch.setattr(runners, "setup_experiment", lambda cfg, **kw: pytest.fail("set up a run"))

        with pytest.raises(ValueError, match="log_train_ranking"):
            runners.run_experiment(cfg)


class TestBuildCallbacks:
    """Early stopping follows the loss's checkpoint metric and stays off while overfitting."""

    def _early_stopping(self, cfg, **overrides) -> list[EarlyStopping]:
        run_cfg = cast(Config, OmegaConf.merge(cfg, overrides))
        loss = _build_bce_loss()
        callbacks = _build_callbacks(loss, run_cfg, run_dir=None, run_name="MLP", debug=False)

        return [callback for callback in callbacks if isinstance(callback, EarlyStopping)]

    def test_early_stopping_monitors_the_checkpoint_metric(self, cfg):
        (early_stopping,) = self._early_stopping(cfg, train={"early_stopping": {"patience": 7}})
        loss = _build_bce_loss()

        assert early_stopping.monitor == loss.monitor
        assert early_stopping.mode == loss.monitor_mode
        assert early_stopping.patience == 7

    def test_null_patience_turns_early_stopping_off(self, cfg):
        assert self._early_stopping(cfg, train={"early_stopping": {"patience": None}}) == []

    def test_overfitting_turns_early_stopping_off(self, cfg):
        assert self._early_stopping(cfg, trainer={"overfit_batches": 1}) == []

    def test_the_best_checkpoint_holds_the_best_epoch_and_the_last_the_final_epoch(self, cfg, tmp_path):
        loss = _build_bce_loss()
        callbacks = _build_callbacks(loss, cfg, run_dir=tmp_path, run_name="MLP", debug=False)
        checkpoints = [callback for callback in callbacks if isinstance(callback, ModelCheckpoint)]
        trainer = L.Trainer(
            max_epochs=4,
            accelerator="cpu",
            logger=False,
            callbacks=checkpoints,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(_WorseningModel())

        checkpoint_dir = tmp_path / CHECKPOINTS_DIR / "MLP"
        assert torch.load(checkpoint_dir / "best.ckpt", weights_only=False)["epoch"] == 0
        assert torch.load(checkpoint_dir / "last.ckpt", weights_only=False)["epoch"] == 3
        assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["best.ckpt", "last.ckpt"]
        assert trainer.checkpoint_callback.best_model_path == str(checkpoint_dir / "best.ckpt")


class TestWandBParams:
    """The experiment name has to reach W&B, or a run cannot be traced back to what produced it."""

    def test_tags_carry_the_dataset_and_the_experiment(self):
        params = WandBParams(use_wandb=True, dataset_name="CMAP", experiment="demographics_ablation")

        assert params.tags() == ["CMAP", "demographics_ablation"]
        assert params.tags("baseline") == ["CMAP", "demographics_ablation", "baseline"]

    def test_unset_fields_are_skipped(self):
        assert WandBParams(use_wandb=False).tags() == []
        assert WandBParams(use_wandb=False, dataset_name="CMAP").tags("baseline") == ["CMAP", "baseline"]


class TestSaveResults:
    def _saved(self, tmp_path, *frames: pl.DataFrame):
        save_results(tmp_path, *frames)
        return pl.read_parquet(tmp_path / RESULTS_FILE), pl.read_parquet(tmp_path / PER_USER_FILE)

    def _written(self, tmp_path) -> list[str]:
        return sorted(f.name for f in tmp_path.iterdir())

    def test_splits_stages_into_two_files(self, tmp_path):
        aggregate, per_user = self._saved(tmp_path, make_model_frame("MLP"), make_model_frame("GATSkip"))

        assert aggregate["stage"].unique().sort().to_list() == ["fit", "test"]
        assert per_user["stage"].unique().to_list() == [PER_USER_STAGE]
        assert per_user.height == 4  # two users per model

    def test_per_user_columns_are_absent_from_the_aggregate_file(self, tmp_path):
        aggregate, per_user = self._saved(tmp_path, make_model_frame("MLP"))

        assert "user_idx" not in aggregate.columns
        assert "val_r_precision" not in per_user.columns
        assert {"name", "seed", "user_idx", "n_pos", "r_precision", "recall"}.issubset(per_user.columns)

    def test_seed_is_retained_for_pairing(self, tmp_path):
        frames = [make_model_frame("MLP", seed=1), make_model_frame("MLP", seed=2)]
        _, per_user = self._saved(tmp_path, *frames)

        assert per_user["seed"].to_list() == [1, 1, 2, 2]
        assert per_user["user_idx"].to_list() == [7, 9, 7, 9]

    def test_a_run_of_baselines_alone_keeps_the_null_seed(self, tmp_path):
        _, per_user = self._saved(tmp_path, make_model_frame("Gravity", seed=None))

        assert per_user["seed"].null_count() == per_user.height

    def test_writes_only_the_aggregate_file_when_no_per_user_rows(self, tmp_path):
        aggregate_only = make_model_frame("MLP").filter(pl.col("stage") != PER_USER_STAGE)
        save_results(tmp_path, aggregate_only)

        assert self._written(tmp_path) == [RESULTS_FILE]

    def test_no_scores_file_when_the_run_saved_no_score_vectors(self, tmp_path):
        save_results(tmp_path, make_model_frame("MLP"))

        assert self._written(tmp_path) == [PER_USER_FILE, RESULTS_FILE]

    def test_home_distances_go_to_their_own_file(self, tmp_path):
        combined = pl.concat([make_model_frame("MLP"), make_home_distance_frame()], how="diagonal")
        save_results(tmp_path, combined)

        assert self._written(tmp_path) == [DISTANCES_FILE, PER_USER_FILE, RESULTS_FILE]

        distances = pl.read_parquet(tmp_path / DISTANCES_FILE)
        assert distances["stage"].unique().to_list() == [HOME_DISTANCE_STAGE]
        assert distances["distances"].to_list() == [[0.0, 900.0, 4000.0], [4000.0, 900.0, 0.0]]

    def test_score_vectors_go_to_their_own_file(self, tmp_path):
        combined = pl.concat([make_model_frame("MLP"), make_score_frame("MLP")], how="diagonal")
        save_results(tmp_path, combined)

        assert self._written(tmp_path) == [PER_USER_FILE, RESULTS_FILE, SCORES_FILE]

        scores = pl.read_parquet(tmp_path / SCORES_FILE)
        assert scores["stage"].unique().to_list() == [SCORE_VECTOR_STAGE]
        assert scores["scores"].to_list() == [[0.1, 0.9, 0.4], [0.7, 0.2, 0.5]]
        assert {"name", "seed", "user_idx"}.issubset(scores.columns)

    def test_fitted_parameters_go_to_their_own_file(self, tmp_path):
        params = _fitted_param_frame({"alpha": -3.0, "mass_jobs": 0.4}, "GravityJobsPop")
        combined = pl.concat([make_model_frame("GravityJobsPop", seed=None), params], how="diagonal")
        save_results(tmp_path, combined)

        assert self._written(tmp_path) == [FITTED_PARAMS_FILE, PER_USER_FILE, RESULTS_FILE]

        saved = pl.read_parquet(tmp_path / FITTED_PARAMS_FILE)
        assert saved["stage"].unique().to_list() == [FITTED_PARAM_STAGE]
        assert saved["parameter"].to_list() == ["alpha", "mass_jobs"]
        assert saved["value"].to_list() == [-3.0, 0.4]
        assert "parameter" not in pl.read_parquet(tmp_path / RESULTS_FILE).columns

    def test_no_fitted_parameters_file_when_no_baseline_reports_any(self, tmp_path):
        params = _fitted_param_frame({}, "Uniform")
        combined = pl.concat([make_model_frame("Uniform", seed=None), params], how="diagonal")
        save_results(tmp_path, combined)

        assert self._written(tmp_path) == [PER_USER_FILE, RESULTS_FILE]

    def test_score_rows_are_absent_from_the_other_two_files(self, tmp_path):
        combined = pl.concat([make_model_frame("MLP"), make_score_frame("MLP")], how="diagonal")
        aggregate, per_user = self._saved(tmp_path, combined)

        assert aggregate["stage"].unique().sort().to_list() == ["fit", "test"]
        assert per_user["stage"].unique().to_list() == [PER_USER_STAGE]
        assert "scores" not in aggregate.columns
        assert "scores" not in per_user.columns


class TestRunRecord:
    """A run directory must record the config used by the experiment, overrides included."""

    def _cfg(self, cfg, tmp_path, **trainer) -> Config:
        overrides = {
            "paths": {"runs": str(tmp_path / "runs"), "overfit": str(tmp_path / "overfit")},
            "trainer": {"fast_dev_run": False, "overfit_batches": 0, **trainer},
        }
        return cast(Config, OmegaConf.merge(cfg, overrides))

    def test_a_run_gets_the_directory_named_by_its_id(self, cfg, tmp_path):
        run_dir = create_output_dir(self._cfg(cfg, tmp_path), RUN)

        assert run_dir == tmp_path / "runs" / RUN
        assert run_dir.is_dir()

    def test_an_overfit_run_gets_its_directory_elsewhere(self, cfg, tmp_path):
        run_dir = create_output_dir(self._cfg(cfg, tmp_path, overfit_batches=1), RUN)

        assert run_dir == tmp_path / "overfit" / RUN
        assert not (tmp_path / "runs").exists()

    def test_a_fast_dev_run_records_nothing(self, cfg, tmp_path):
        assert create_output_dir(self._cfg(cfg, tmp_path, fast_dev_run=True), RUN) is None
        assert list(tmp_path.iterdir()) == []

    def test_the_saved_config_holds_the_architectures_as_trained(self, cfg, tmp_path):
        models = {**cfg.models, "gat": with_overrides(cfg.models.gat, {"lr": 0.5})}
        save_config(tmp_path, cfg, models)

        saved = OmegaConf.load(tmp_path / CONFIG_FILE)
        assert saved.models.gat.lr == 0.5
        assert saved.models.mlp.lr == cfg.models.mlp.lr
        assert saved.data.name == cfg.data.name

    def test_the_wandb_config_leaves_out_the_architectures(self, cfg):
        config = run_config(cfg)

        assert "models" not in config
        assert config["train"]["loss"]["type"] == cfg.train.loss.type


class _WorseningModel(BoringModel):
    """A model whose logged ``val_bce`` rises every epoch, so its best epoch is the first."""

    def validation_step(self, batch, batch_idx):
        self.log("val_bce", float(self.current_epoch))


def source_run_id(name: str) -> str:
    """The id of the recorded run of the dataset ``name`` that the restored variants come from."""
    return f"{name}-edge_ablation-20261117-183417"


def write_source_run(tmp_path, compose_config, *overrides: str):
    """Record the config of a run of the composed config under ``tmp_path/runs``; return the runs path and config."""
    cfg = compose_config(*overrides)
    run_path = tmp_path / "runs" / source_run_id(cfg.data.name)
    run_path.mkdir(parents=True)
    OmegaConf.save(cfg, run_path / CONFIG_FILE, resolve=True)

    return tmp_path / "runs", cfg


def write_checkpoint(run_path, run_name: str, scorer: torch.nn.Module) -> None:
    """Write ``scorer``'s weights to ``checkpoints/{run_name}/best.ckpt`` of ``run_path``, as Lightning names them."""
    checkpoint_dir = run_path / CHECKPOINTS_DIR / run_name
    checkpoint_dir.mkdir(parents=True)
    state_dict = {f"scorer.{key}": value for key, value in scorer.state_dict().items()}
    torch.save({"state_dict": state_dict}, checkpoint_dir / "best.ckpt")


def read_source(tmp_path, compose_config, *overrides: str) -> SourceRun:
    """Record a run of the ``edge_ablation`` experiment under ``tmp_path`` and read it back."""
    runs_path, cfg = write_source_run(tmp_path, compose_config, "experiment=edge_ablation", *overrides)

    return SourceRun.read(runs_path, source_run_id(cfg.data.name))


def transfer_config(cfg: Config, **settings) -> TransferConfig:
    """The transfer settings of ``cfg`` with ``settings`` merged in."""
    return cast(TransferConfig, OmegaConf.merge(cfg.transfer, settings))


def restore(source: SourceRun, dataset_name: str = "TTS2022", **settings) -> list[RestoredVariant]:
    """The variants restored from ``source`` in an experiment on ``dataset_name``, with ``settings`` as the transfer."""
    transfer = transfer_config(source.cfg, source_run=source.run, **settings)

    return build_restored_variants(transfer, source, dataset_name)


def restored_names(source: SourceRun, dataset_name: str = "TTS2022", **settings) -> list[str]:
    """Names of the variants restored from ``source`` in an experiment on ``dataset_name``."""
    return [restored_variant.name for restored_variant in restore(source, dataset_name, **settings)]


class TestRestoredVariantNames:
    def test_a_variant_restored_from_another_dataset_names_it(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        assert restored_names(source, variants=["MLP"]) == ["MLP-fromTPG"]

    def test_the_target_scaling_is_named(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        assert restored_names(source, variants=["MLP"], scalings=["source", "target"]) == [
            "MLP-fromTPG",
            "MLP-fromTPG-tgtscale",
        ]

    def test_the_tested_copies_come_before_the_fine_tuned_ones(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        names = restored_names(source, variants=["MLP", "GATv2-2-norail"], fine_tune=True)

        assert names == ["MLP-fromTPG", "MLP-fromTPG-ft", "GATv2-2-norail-fromTPG", "GATv2-2-norail-fromTPG-ft"]

    def test_a_variant_can_be_fine_tuned_only(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        assert restored_names(source, variants=["MLP"], test_restored=False, fine_tune=True) == ["MLP-fromTPG-ft"]

    def test_a_variant_restored_from_its_own_dataset_keeps_its_name(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        assert restored_names(source, source.dataset_name, variants=["MLP"]) == ["MLP"]

    def test_a_variant_fine_tuned_on_its_own_dataset_is_named_ft(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        names = restored_names(source, source.dataset_name, variants=["MLP"], test_restored=False, fine_tune=True)

        assert names == ["MLP-ft"]


class TestBuildRestoredVariants:
    def test_keeps_the_model_config_of_the_recorded_run(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        (restored_variant,) = restore(source, variants=["GATv2-2-norail"])

        assert restored_variant.variant.model.num_gcn_layers == 2
        assert restored_variant.variant.features == InputFeatures(use_rail_edges=False)

    def test_a_variant_the_run_did_not_train_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(KeyError, match="MLP-nott"):
            restore(source, variants=["MLP-nott"])

    def test_an_unknown_scaling_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(ValueError, match="scaled by one of"):
            restore(source, variants=["MLP"], scalings=["other"])

    def test_the_target_scaling_of_the_own_dataset_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(ValueError, match="`source` scaling alone"):
            restore(source, source.dataset_name, variants=["MLP"], scalings=["source", "target"])

    def test_variants_neither_tested_nor_fine_tuned_raise(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(ValueError, match="tested, fine-tuned or both"):
            restore(source, variants=["MLP"], test_restored=False)

    def test_logs_the_restore_settings(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        (restored_variant,) = restore(source, variants=["MLP"], scalings=["target"])

        assert restored_variant.hyperparams["restore"] == {
            "source_run": source.run,
            "scaling": "target",
            "fine_tune": False,
        }


class TestReadSourceRun:
    def test_no_recorded_run_restores_nothing(self, cfg, tmp_path):
        assert read_source_run(tmp_path, cfg.transfer) is None

    @pytest.mark.parametrize("settings", [{"variants": ["MLP"]}, {"fit_baselines_on_source": True}])
    def test_settings_that_read_the_run_need_one(self, cfg, tmp_path, settings):
        with pytest.raises(ValueError, match="source_run"):
            read_source_run(tmp_path, transfer_config(cfg, **settings))

    def test_reads_the_run_of_its_id(self, tmp_path, compose_config):
        runs_path, cfg = write_source_run(tmp_path, compose_config, "experiment=edge_ablation")
        transfer = transfer_config(cfg, source_run=source_run_id(cfg.data.name))
        source = read_source_run(runs_path, transfer)

        assert source is not None
        assert source.run == source_run_id(cfg.data.name)


class TestSourceRun:
    def test_reads_the_checkpoint_of_each_seed(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        write_checkpoint(source.path, "MLP-seed43", torch.nn.Linear(1, 1))

        assert source.checkpoint_path("MLP", 43) == source.path / CHECKPOINTS_DIR / "MLP-seed43" / "best.ckpt"

    def test_a_seed_the_run_did_not_train_with_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(ValueError, match="seeds"):
            source.checkpoint_path("MLP", 7)

    def test_a_missing_checkpoint_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        with pytest.raises(FileNotFoundError, match="MLP"):
            source.checkpoint_path("MLP", 42)

    def test_a_single_seed_run_names_the_checkpoint_after_the_variant(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config, "train.train_seeds=[5]")
        write_checkpoint(source.path, "MLP", torch.nn.Linear(1, 1))

        assert source.checkpoint_path("MLP", 5).parent.name == "MLP"


class TestCheckCheckpoints:
    def test_checks_every_seed_of_every_restored_variant(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        write_checkpoint(source.path, "MLP-seed42", torch.nn.Linear(1, 1))
        restored_variants = restore(source, variants=["MLP"])

        with pytest.raises(FileNotFoundError, match="seed 43"):
            check_checkpoints(restored_variants, [42, 43])

    def test_the_fine_tuned_copies_read_the_checkpoint_of_the_variant(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        write_checkpoint(source.path, "MLP-seed42", torch.nn.Linear(1, 1))

        check_checkpoints(restore(source, variants=["MLP"], fine_tune=True), [42])


class TestRestoreScorer:
    def test_restores_the_weights_of_the_checkpoint(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config, "train.train_seeds=[42]")
        (restored_variant,) = restore(source, variants=["MLP"])
        locations, _ = make_survey()
        torch.manual_seed(0)
        trained_scorer = build_scorer(
            source.cfg.models.mlp, DEFAULT_FEATURES, locations, make_schema(), source.cfg.train.home_pe
        )
        write_checkpoint(source.path, "MLP", trained_scorer)

        torch.manual_seed(1)
        scorer = restore_scorer(restored_variant, 42, locations, make_schema())

        for key, value in trained_scorer.state_dict().items():
            assert torch.equal(scorer.state_dict()[key], value)

    def test_a_checkpoint_of_other_widths_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config, "train.train_seeds=[42]")
        (restored_variant,) = restore(source, variants=["MLP"])
        locations, _ = make_survey()
        trained_scorer = build_scorer(
            source.cfg.models.mlp, DEFAULT_FEATURES, locations, make_schema(), source.cfg.train.home_pe
        )
        write_checkpoint(source.path, "MLP", trained_scorer)
        wider_schema = make_schema(num_location_features=7)

        with pytest.raises(RuntimeError, match="size mismatch"):
            restore_scorer(restored_variant, 42, locations, wider_schema)


class TestCheckSameSchema:
    def test_equal_schemas_pass(self):
        _check_same_schema(make_schema(), make_schema(), "TTS2016", "TTS2022")

    def test_names_the_groups_that_differ_and_their_columns(self):
        with pytest.raises(ValueError, match=r"`locations`: \[.*\] in TTS2016, \[.*'c6'\] in CMAP") as error:
            _check_same_schema(make_schema(), make_schema(num_location_features=7), "TTS2016", "CMAP")

        assert "user_attributes" not in str(error.value)


class TestCheckTestedAcrossDatasets:
    def test_variants_of_another_dataset_only_tested_pass(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        restored_variants = restore(source, variants=["MLP"], scalings=["source", "target"])
        cfg = compose_config("data=tts2022")

        _check_tested_across_datasets(cfg, [], restored_variants, source)

    def test_a_variant_trained_from_scratch_raises(self, cfg):
        with pytest.raises(ValueError, match="only tested"):
            _check_tested_across_datasets(cfg, [Variant(cfg.models.mlp)], [], None)

    def test_baselines_fitted_on_a_run_of_this_dataset_raise(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        cfg = compose_config("transfer.fit_baselines_on_source=true")

        with pytest.raises(ValueError, match="baselines fitted on TPG"):
            _check_tested_across_datasets(cfg, [], [], source)

    def test_baselines_fitted_on_a_run_of_another_dataset_pass(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        cfg = compose_config("data=tts2022", "transfer.fit_baselines_on_source=true")

        _check_tested_across_datasets(cfg, [], [], source)

    @pytest.mark.parametrize(
        ("dataset_name", "settings"),
        [("TTS2022", {"fine_tune": True}), ("TPG", {})],
        ids=["fine-tuned", "own-dataset"],
    )
    def test_a_restored_variant_trained_on_this_dataset_raises(self, tmp_path, compose_config, dataset_name, settings):
        source = read_source(tmp_path, compose_config)
        restored_variants = restore(source, dataset_name, variants=["MLP"], **settings)

        cfg = compose_config()

        with pytest.raises(ValueError, match="only tested"):
            _check_tested_across_datasets(cfg, [], restored_variants, source)


class TestCheckSameSplit:
    def test_the_split_of_the_recorded_run_passes(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)

        check_same_split(source, source.split, source.cfg.train.split_seed)

    def test_another_split_seed_raises(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        other_seed = source.cfg.train.split_seed + 1

        with pytest.raises(ValueError, match="trained on"):
            check_same_split(source, source.split, other_seed)

    def test_other_split_fractions_raise(self, tmp_path, compose_config):
        source = read_source(tmp_path, compose_config)
        other_split = HouseholdSplit(val_size=source.split.val_size, test_size=0.5)

        with pytest.raises(ValueError, match="trained on"):
            check_same_split(source, other_split, source.cfg.train.split_seed)


class TestCheckNodeWeights:
    def _locations(self, loc_ids: list[int]):
        return types.SimpleNamespace(node_ids=pl.DataFrame({"loc_id": loc_ids}))

    def _source_nodes(self, monkeypatch, loc_ids: list[int]) -> None:
        source_node_ids = pl.DataFrame({"loc_id": loc_ids})
        monkeypatch.setattr(SourceRun, "node_ids", lambda self: source_node_ids)

    def test_a_node_bias_of_reordered_nodes_raises(self, tmp_path, compose_config, monkeypatch):
        source = read_source(tmp_path, compose_config, "models.two_tower.bias=node")
        restored_variants = restore(source, variants=["TwoTower-4-nodebias"])
        self._source_nodes(monkeypatch, [2, 1, 3])

        with pytest.raises(ValueError, match="TwoTower-4-nodebias-fromTPG"):
            check_node_weights(restored_variants, self._locations([1, 2, 3]))  # type: ignore[arg-type]

    def test_a_node_bias_of_the_same_nodes_passes(self, tmp_path, compose_config, monkeypatch):
        source = read_source(tmp_path, compose_config, "models.two_tower.bias=node")
        restored_variants = restore(source, variants=["TwoTower-4-nodebias"])
        self._source_nodes(monkeypatch, [1, 2, 3])

        check_node_weights(restored_variants, self._locations([1, 2, 3]))  # type: ignore[arg-type]

    def test_a_variant_with_no_node_weights_takes_other_nodes(self, tmp_path, compose_config, monkeypatch):
        source = read_source(tmp_path, compose_config)
        restored_variants = restore(source, variants=["MLP", "TwoTower-4"])
        self._source_nodes(monkeypatch, [7, 8])

        check_node_weights(restored_variants, self._locations([1, 2, 3]))  # type: ignore[arg-type]


class TestHasNodeWeights:
    @pytest.mark.parametrize(
        ("key", "overrides", "expected"),
        [
            ("mlp", {}, False),
            ("gat", {}, False),
            ("two_tower", {}, False),
            ("two_tower", {"bias": "node"}, True),
            ("two_tower", {"node_embedding": True}, True),
            ("home_zone_mf", {}, True),
        ],
    )
    def test_flags_the_models_with_one_weight_per_node(self, cfg, key, overrides, expected):
        model = with_overrides(cfg.models[key], overrides)

        assert has_node_weights(model) is expected


TRANSFER_RESTORED_NAMES = {
    "transfer_tts2022": [
        f"{name}-fromTTS2016{scaling}"
        for name in [
            "MLP",
            "GATv2-2",
            "GATv2-2-norail",
            "GATv2-2-nocontig-norail",
            "TwoTower-4",
            "TwoTower-4-norail",
            "TwoTower-4-nocontig-norail",
        ]
        for scaling in ["", "-tgtscale"]
    ],
    "transfer_finetune": [
        "MLP-fromTTS2016",
        "MLP-fromTTS2016-ft",
        "TwoTower-4-fromTTS2016",
        "TwoTower-4-fromTTS2016-ft",
    ],
}
TRANSFER_RESTORED_NAMES["transfer_all_users"] = TRANSFER_RESTORED_NAMES["transfer_tts2022"]


class TestTransferExperiments:
    @pytest.mark.parametrize("experiment", sorted(TRANSFER_RESTORED_NAMES))
    def test_restores_variants_trained_by_the_recorded_run(self, tmp_path, compose_config, experiment):
        runs_path, _ = write_source_run(tmp_path, compose_config, "data=tts2016", "experiment=edge_ablation")
        cfg = compose_config("data=tts2022", f"experiment={experiment}")
        source = read_source_run(runs_path, cfg.transfer)
        assert source is not None

        restored_variants = build_restored_variants(cfg.transfer, source, cfg.data.name)

        assert [restored_variant.name for restored_variant in restored_variants] == TRANSFER_RESTORED_NAMES[experiment]


class TestRunRestoredVariants:
    def _setup(self, cfg: Config, restored_variants: list[RestoredVariant]) -> ExperimentSetup:
        setup = make_setup(cfg, [42], [])
        datamodules = {
            "source": types.SimpleNamespace(locations="source", schema=None),
            "target": types.SimpleNamespace(locations="target", schema=None),
        }

        return dataclasses.replace(setup, restored_variants=restored_variants, scaled_datamodules=datamodules)  # type: ignore[arg-type]

    def _restored_variant(self, cfg: Config, scaling: str, fine_tune: bool) -> RestoredVariant:
        source = SourceRun(Path(source_run_id(cfg.data.name)), cfg, {})

        return RestoredVariant(source, Variant(cfg.models.mlp), scaling, fine_tune, across_datasets=True)

    def _record(self, monkeypatch) -> list[dict]:
        """Stub the restoring, logging, training and testing of a restored variant; return one record per model."""
        calls = record_training(monkeypatch)
        monkeypatch.setattr(runners, "restore_scorer", lambda restored_variant, seed, locations, schema: locations)

        return calls

    def test_a_restored_variant_is_tested_on_the_datamodule_of_its_scaling(self, cfg, monkeypatch):
        calls = self._record(monkeypatch)
        restored_variants = [self._restored_variant(cfg, "source", False), self._restored_variant(cfg, "target", False)]
        _run_variants(self._setup(cfg, restored_variants))

        assert [call["scorer"] for call in calls] == ["source", "target"]
        assert [call["model"] for call in calls] == [None, None]
        assert [call["hyperparams"]["model_type"] for call in calls] == ["restored", "restored"]

    def test_a_fine_tuned_variant_trains_at_the_scaled_learning_rate_then_is_tested(self, cfg, monkeypatch):
        calls = self._record(monkeypatch)
        _run_variants(self._setup(cfg, [self._restored_variant(cfg, "source", True)]))

        (call,) = calls
        assert call["name"] == "MLP-fromTPG-ft"
        assert call["model"].lr == pytest.approx(cfg.models.mlp.lr * cfg.transfer.fine_tune_lr_scale)
        assert call["hyperparams"]["model_type"] == "fine_tuned"


class _SurveyStub(L.LightningDataModule):
    """The two users of ``make_survey``, each in the training, validation and test splits."""

    def __init__(self):
        super().__init__()
        self.locations, users = make_survey()
        self.schema = make_schema()
        self.dataset = UserDataset(users, [0, 1])
        self.home_coverage = _compute_home_coverage(self.dataset, self.dataset)

    def setup(self, stage: str | None = None) -> None:
        """Hold the users already."""

    def train_dataloader(self) -> DataLoader:
        """Return a loader over both users."""
        return DataLoader(self.dataset, batch_size=2)

    def val_dataloader(self) -> DataLoader:
        """Return a loader over both users."""
        return DataLoader(self.dataset, batch_size=2)

    def test_dataloader(self) -> DataLoader:
        """Return a loader over both users."""
        return DataLoader(self.dataset, batch_size=2)


class _HyperparamsLogger(Logger):
    """Keeps every set of hyperparameters logged to it."""

    def __init__(self):
        super().__init__()
        self.logged_hyperparams: list = []

    @property
    def name(self) -> str:
        """Name of the logger."""
        return "hyperparams"

    @property
    def version(self) -> int:
        """Version of the logger."""
        return 0

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        """Keep ``params``."""
        self.logged_hyperparams.append(params)

    def log_metrics(self, metrics, step=None) -> None:
        """Ignore the metrics."""


class TestTrainAndEvaluateModel:
    def _cfg(self, cfg: Config) -> Config:
        overrides = {
            "trainer": {
                "max_epochs": 2,
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "limit_train_batches": 1.0,
            },
            "train": {"compile": False, "schedule_lr": False, "early_stopping": {"patience": None}},
        }

        return cast(Config, OmegaConf.merge(cfg, overrides))

    def _scorer(self, cfg: Config, datamodule: _SurveyStub):
        return build_scorer(cfg.models.mlp, DEFAULT_FEATURES, datamodule.locations, datamodule.schema, HOME_PE)

    def _evaluate(self, cfg: Config, fit_rows: list[dict] | None, logger: Logger | None = None) -> pl.DataFrame:
        cfg = self._cfg(cfg)
        datamodule = _SurveyStub()
        loss = build_loss(cfg.train.loss, cfg.train.max_recall_k)
        scorer = self._scorer(cfg, datamodule)

        return evaluate_model(scorer, "MLP", datamodule, loss, cfg, logger, fit_rows)  # type: ignore[arg-type]

    def test_testing_logs_no_hyperparameters_over_those_of_the_training(self, cfg):
        logger = _HyperparamsLogger()
        self._evaluate(cfg, fit_rows=None, logger=logger)

        assert logger.logged_hyperparams == []

    def test_an_untrained_scorer_is_validated_once_then_tested(self, cfg):
        results = self._evaluate(cfg, fit_rows=None)
        aggregate = results.filter(pl.col("stage").is_in(["fit", "test"]))

        assert aggregate["stage"].to_list() == ["fit", "test"]
        assert aggregate["epoch"].to_list() == [0, None]
        assert results.filter(pl.col("stage") == PER_USER_STAGE).height == 2

    def test_the_epoch_rows_of_a_trained_scorer_replace_its_validation(self, cfg):
        fit_rows = [{"stage": "fit", "epoch": 3, "train_loss": 0.5}]
        results = self._evaluate(cfg, fit_rows)
        fit = results.filter(pl.col("stage") == "fit")

        assert fit["epoch"].to_list() == [3]
        assert fit["train_loss"].to_list() == [0.5]

    def test_training_loads_the_best_checkpoint_into_the_scorer(self, cfg, tmp_path, monkeypatch):
        cfg = self._cfg(cfg)
        datamodule = _SurveyStub()
        loss = build_loss(cfg.train.loss, cfg.train.max_recall_k)
        scorer = self._scorer(cfg, datamodule)
        loaded_checkpoints: list[tuple] = []
        monkeypatch.setattr(
            training, "load_scorer_weights", lambda scorer, path: loaded_checkpoints.append((scorer, path))
        )

        fit_rows = train_model(scorer, "MLP", cfg.models.mlp, datamodule, loss, cfg, tmp_path, None, False)  # type: ignore[arg-type]

        assert [row["epoch"] for row in fit_rows] == [0, 1]
        assert loaded_checkpoints == [(scorer, tmp_path / CHECKPOINTS_DIR / "MLP" / "best.ckpt")]

    def test_training_with_no_run_directory_writes_and_loads_no_checkpoint(self, cfg, tmp_path, monkeypatch):
        cfg = self._cfg(cfg)
        datamodule = _SurveyStub()
        loss = build_loss(cfg.train.loss, cfg.train.max_recall_k)
        loaded_checkpoints: list[tuple] = []
        monkeypatch.setattr(
            training, "load_scorer_weights", lambda scorer, path: loaded_checkpoints.append((scorer, path))
        )
        monkeypatch.chdir(tmp_path)

        train_model(self._scorer(cfg, datamodule), "MLP", cfg.models.mlp, datamodule, loss, cfg, None, None, False)  # type: ignore[arg-type]

        assert loaded_checkpoints == []


def test_two_variants_of_one_name_raise():
    with pytest.raises(ValueError, match="repeat"):
        check_distinct_names(["MLP", "MLP-fromTPG", "MLP"])


def test_only_uniform_and_distance_decay_read_no_visit():
    specs = _closed_form_baselines(num_nodes=4, travel_time_schema=TRAVEL_TIME_SCHEMA)
    visit_free_names = [spec.name for spec in specs if not spec.reads_visits]

    assert visit_free_names == ["Uniform", "DistanceDecay"]
