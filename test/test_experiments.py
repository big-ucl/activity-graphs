"""Tests for the variants, the seed loop and the aggregate / per-user result split."""

import dataclasses
import inspect
import types
from datetime import datetime
from typing import cast

import lightning as L
import polars as pl
import pytest
import torch
from encoders import continuous_schema
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from activitygraphs.base import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config, HomePEConfig, read_config
from activitygraphs.experiments import runners
from activitygraphs.experiments.results import (
    _drop_empty_columns,
    create_output_dir,
    run_config,
    save_config,
    save_results,
)
from activitygraphs.experiments.runners import _run_variants
from activitygraphs.experiments.setup import ExperimentSetup
from activitygraphs.experiments.training import WandBParams, _build_callbacks, _trainer_arguments
from activitygraphs.experiments.variants import (
    FEATURE_SUFFIXES,
    Variant,
    build_variants,
    variant_hyperparams,
    with_overrides,
)
from activitygraphs.models.architectures import BUILDERS, _lookup_architecture, build_scorer
from activitygraphs.models.baselines import HomeZoneMFBaseline
from activitygraphs.models.features import (
    AccessFeatures,
    InputFeatures,
    LocationFeatures,
    RandomWalkPE,
    build_feature_encoder,
    build_node_feature_encoder,
)
from activitygraphs.models.networks import GAT, GraphTransformer, NodeMLP
from activitygraphs.models.scorers import GraphScorer, NodeScorer
from activitygraphs.models.two_tower import TwoTowerScorer
from activitygraphs.runs import CHECKPOINTS_DIR, CONFIG_FILE, DISTANCES_FILE, PER_USER_FILE, RESULTS_FILE, SCORES_FILE
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors.build import USER_NODE_SCHEMA
from activitygraphs.training.losses import BCELoss
from activitygraphs.utils import get_project_root

STARTED = datetime(2026, 10, 26, 15, 30, 0)
EXPERIMENT_CONFIGS = sorted((get_project_root() / "activitygraphs/conf/experiment").glob("*.yaml"))
HOME_PE = HomePEConfig(bins=8, max_hops=40)
DEFAULT_FEATURES = InputFeatures()


@pytest.fixture(scope="module")
def cfg() -> Config:
    return read_config(get_project_root(), verbose=False)


def make_setup(cfg: Config, train_seeds: list[int]) -> ExperimentSetup:
    """An ExperimentSetup with no datamodule; ``build_scorer`` is stubbed by the tests, so no data is needed."""
    return ExperimentSetup(
        cfg=cfg,
        run_dir=None,
        datamodule=types.SimpleNamespace(locations=None, schema=None),  # type: ignore[arg-type]
        loss=None,  # type: ignore[arg-type]
        models=dict(cfg.models),
        baseline_results=[],
        home_distances=pl.DataFrame(),
        train_seeds=train_seeds,
        debug=False,
        wandb_params=WandBParams(use_wandb=False),
    )


def make_schema(
    num_location_features: int = 5, num_attributes: int = 3, num_edge_features: int = 2, num_access_features: int = 4
) -> DatasetSchema:
    """The schema of a survey with continuous location, access, edge and user attribute columns."""
    return DatasetSchema(
        locations=continuous_schema("locations", num_location_features),
        access=continuous_schema("access", num_access_features),
        edges=continuous_schema("edges", num_edge_features),
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=continuous_schema("user_attributes", num_attributes),
        distances=continuous_schema("distances", 1),
        targets=continuous_schema("targets", 1),
    )


def make_locations() -> types.SimpleNamespace:
    """Stand-in exposing only what the scorer builders read off ``Locations``."""
    return types.SimpleNamespace(num_rwpe_features=2, num_nodes=6)


def record_training(monkeypatch, frame=None) -> list[dict]:
    """Replace scorer building and training with stubs, returning the list each training call is recorded into."""
    calls: list[dict] = []

    def train_and_evaluate_model(**kwargs):
        calls.append(kwargs)
        name = kwargs["variant"].name
        return frame(name) if frame else pl.DataFrame({"name": [name], "stage": ["test"]})

    monkeypatch.setattr(runners, "train_and_evaluate_model", train_and_evaluate_model)
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
        with pytest.raises(ConfigKeyError, match="sweeps"):
            variant_names(cfg, {"model": "mlp", "sweeps": {}})

    def test_an_unknown_flag_raises(self, cfg):
        with pytest.raises(TypeError, match="use_hpe"):
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

    def test_input_width_follows_the_variant_features(self, cfg):
        locations = make_locations()
        schema = make_schema(num_location_features=5, num_attributes=3, num_access_features=4)
        plain_features = InputFeatures(use_home_pe=False, use_dist=False, use_demographics=False)
        plain = build_feature_encoder(plain_features, locations, schema, HOME_PE)  # type: ignore[arg-type]
        dist = build_feature_encoder(InputFeatures(use_home_pe=False), locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert plain.out_dim == 5 + 4 + 1 + 2
        assert dist.out_dim == 5 + 4 + 1 + 2 + 3 + 1

    def test_rwpe_free_input_width_leaves_out_the_encoding(self, cfg):
        locations = make_locations()
        schema = make_schema(num_location_features=5, num_attributes=3, num_access_features=4)
        features = InputFeatures(use_home_pe=False, use_rwpe=False)
        encoder = build_feature_encoder(features, locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert encoder.out_dim == 5 + 4 + 1 + 3 + 1

    def test_the_access_flag_adds_the_access_columns(self, cfg):
        locations, schema = make_locations(), make_schema(num_access_features=4)
        without_access = build_feature_encoder(InputFeatures(use_access=False), locations, schema, HOME_PE)  # type: ignore[arg-type]
        with_access = build_feature_encoder(InputFeatures(), locations, schema, HOME_PE)  # type: ignore[arg-type]

        assert with_access.out_dim - without_access.out_dim == 4

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

    def test_the_network_takes_the_encoder_width(self, cfg):
        scorer = scorer_of(cfg.models.mlp)

        assert isinstance(scorer, NodeScorer) and isinstance(scorer.network, NodeMLP)
        assert scorer.network.lins[0].in_features == scorer.encoder.out_dim

    def test_edge_dim_reaches_the_networks_that_take_it(self, cfg):
        schema = make_schema(num_edge_features=2)
        gat = scorer_of(cfg.models.gat, schema=schema)
        gps = scorer_of(cfg.models.gps, schema=schema)

        assert isinstance(gat, GraphScorer) and isinstance(gps, GraphScorer)
        assert isinstance(gat.network, GAT) and isinstance(gps.network, GraphTransformer)
        assert gps.network.edge_proj.in_features == 2

    def test_gat_norm_follows_the_config(self, cfg):
        normed = scorer_of(cfg.models.gat)
        plain = scorer_of(with_overrides(cfg.models.gat, {"norm": None}))

        assert isinstance(normed, GraphScorer) and isinstance(plain, GraphScorer)
        assert all(isinstance(norm, torch.nn.LayerNorm) for norm in normed.network.convs.norms)  # type: ignore[union-attr]
        assert all(isinstance(norm, torch.nn.Identity) for norm in plain.network.convs.norms)  # type: ignore[union-attr]

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

        with pytest.raises(KeyError, match="unknown"):
            _lookup_architecture(model)


class TestRunVariants:
    def _variants(self, cfg) -> list[Variant]:
        return [Variant(cfg.models.mlp), Variant(cfg.models.gat, InputFeatures(use_home_pe=False))]

    def test_trains_every_variant_once_per_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        frames = _run_variants(make_setup(cfg, [1, 2]), self._variants(cfg))

        assert len(frames) == 4
        assert [call["variant"].name for call in calls] == ["MLP", "GATv2-8-nohpe", "MLP", "GATv2-8-nohpe"]

    def test_tags_each_frame_with_its_training_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        seeds = pl.concat(_run_variants(make_setup(cfg, [1, 2]), self._variants(cfg)))["seed"].to_list()

        assert seeds == [1, 1, 2, 2]

    def test_builds_a_fresh_model_for_every_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        built: list[torch.nn.Module] = []

        def build(config, features, locations, schema, home_pe_cfg):
            built.append(torch.nn.Linear(1, 1))
            return built[-1]

        monkeypatch.setattr(runners, "build_scorer", build)
        _run_variants(make_setup(cfg, [1, 2]), [Variant(cfg.models.mlp)])

        assert len(built) == 2
        assert built[0] is not built[1]

    def test_every_variant_starts_from_the_seeded_random_state(self, cfg, monkeypatch):
        record_training(monkeypatch)
        draws: list[float] = []

        def build(config, features, locations, schema, home_pe_cfg):
            draws.append(torch.rand(1).item())
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(runners, "build_scorer", build)
        _run_variants(make_setup(cfg, [1]), self._variants(cfg))

        assert draws[0] == draws[1]

    def test_run_tag_separates_checkpoints_across_seeds(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [1, 2]), self._variants(cfg))

        assert [call["run_tag"] for call in calls] == ["seed1", "seed1", "seed2", "seed2"]

    def test_no_run_tag_for_a_single_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [42]), self._variants(cfg))

        assert [call["run_tag"] for call in calls] == [None, None]

    def test_forwards_each_variant(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        _run_variants(make_setup(cfg, [42]), self._variants(cfg))

        assert [call["variant"] for call in calls] == self._variants(cfg)


class TestVariantHyperparams:
    def test_logs_the_architecture_and_flags(self, cfg):
        hyperparams = variant_hyperparams(Variant(cfg.models.gat, InputFeatures(use_dist=False)))

        assert hyperparams["architecture"]["num_gcn_layers"] == cfg.models.gat.num_gcn_layers
        assert hyperparams["use_dist"] is False


class TestDropEmptyColumns:
    def test_drops_all_null_columns_only(self):
        df = pl.DataFrame({"kept": [1, None], "dropped": [None, None]})

        assert _drop_empty_columns(df).columns == ["kept"]

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
            return make_setup(cfg, [42, 43])

        monkeypatch.setattr(runners, "setup_experiment", setup_experiment)
        monkeypatch.setattr(runners, "save_results", lambda *a, **kw: None)

        runners.run_experiment(cfg)

        return calls, baselines_fitted

    def test_trains_every_variant_once_per_seed_in_order(self, compose_config, monkeypatch):
        calls, _ = self._run(compose_config, monkeypatch, "experiment=headline")

        assert [call["variant"].name for call in calls] == ["MLP", "GATv2-2", "TwoTower-4", "HomeZoneMF"] * 2

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
        callbacks = _build_callbacks(BCELoss, run_cfg, run_dir=None, run_name="MLP", debug=False)

        return [callback for callback in callbacks if isinstance(callback, EarlyStopping)]

    def test_early_stopping_monitors_the_checkpoint_metric(self, cfg):
        (early_stopping,) = self._early_stopping(cfg, train={"early_stopping": {"patience": 7}})

        assert early_stopping.monitor == BCELoss.monitor
        assert early_stopping.mode == BCELoss.monitor_mode
        assert early_stopping.patience == 7

    def test_null_patience_turns_early_stopping_off(self, cfg):
        assert self._early_stopping(cfg, train={"early_stopping": {"patience": None}}) == []

    def test_overfitting_turns_early_stopping_off(self, cfg):
        assert self._early_stopping(cfg, trainer={"overfit_batches": 1}) == []

    def test_the_best_checkpoint_holds_the_best_epoch_and_the_last_the_final_epoch(self, cfg, tmp_path):
        callbacks = _build_callbacks(BCELoss, cfg, run_dir=tmp_path, run_name="MLP", debug=False)
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

    def test_a_run_gets_a_numbered_directory(self, cfg, tmp_path):
        run_dir = create_output_dir(self._cfg(cfg, tmp_path), STARTED)

        assert run_dir == tmp_path / "runs" / f"{cfg.data.name}-1-20261026-153000"
        assert run_dir.is_dir()

    def test_an_overfit_run_gets_an_unnumbered_directory_elsewhere(self, cfg, tmp_path):
        run_dir = create_output_dir(self._cfg(cfg, tmp_path, overfit_batches=1), STARTED)

        assert run_dir == tmp_path / "overfit" / f"{cfg.data.name}-20261026-153000"
        assert not (tmp_path / "runs").exists()

    def test_a_fast_dev_run_records_nothing(self, cfg, tmp_path):
        assert create_output_dir(self._cfg(cfg, tmp_path, fast_dev_run=True), STARTED) is None
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
