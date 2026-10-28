"""Tests for the variants, the seed loop and the aggregate / per-user result split."""

import inspect
import types
from datetime import datetime
from typing import cast

import lightning as L
import polars as pl
import pytest
import torch
from encoders import continuous_schema
from omegaconf import OmegaConf

from activitygraphs.base import HOME_DISTANCE_STAGE, PER_USER_STAGE, SCORE_VECTOR_STAGE
from activitygraphs.config import Config, HomePEConfig, read_config
from activitygraphs.experiments import runners
from activitygraphs.experiments.architectures import BUILDERS, architecture_builder
from activitygraphs.experiments.results import (
    create_output_dir,
    drop_empty_columns,
    run_config,
    save_config,
    save_results,
)
from activitygraphs.experiments.runners import run_variants
from activitygraphs.experiments.setup import ExperimentSetup, with_overrides
from activitygraphs.experiments.training import WandBParams
from activitygraphs.experiments.variants import Variant, build_feature_encoder, build_model, variant_hyperparams
from activitygraphs.main import EXPERIMENTS
from activitygraphs.ml.models import GAT, GraphTransformer, NodeMLP
from activitygraphs.runs import CONFIG_FILE, DISTANCES_FILE, PER_USER_FILE, RESULTS_FILE, SCORES_FILE
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors.build import USER_NODE_SCHEMA
from activitygraphs.utils import get_project_root

STARTED = datetime(2026, 10, 26, 15, 30, 0)
EXPERIMENT_CONFIGS = sorted((get_project_root() / "activitygraphs/conf/experiment").glob("*.yaml"))
HOME_PE = HomePEConfig(bins=8, max_hops=40)
IN_CHANNELS = 10


@pytest.fixture(scope="module")
def cfg() -> Config:
    return read_config(get_project_root(), verbose=False)


def make_setup(cfg: Config, train_seeds: list[int]) -> ExperimentSetup:
    """An ExperimentSetup with no datamodule; ``build_feature_encoder`` and ``build_model`` are stubbed by the tests, so no data is needed."""
    return ExperimentSetup(
        cfg=cfg,
        run_dir=None,
        datamodule=types.SimpleNamespace(train_dataset=None),  # type: ignore[arg-type]
        loss=None,  # type: ignore[arg-type]
        models=dict(cfg.models),
        baseline_results=[],
        home_distances=pl.DataFrame(),
        train_seeds=train_seeds,
        debug=False,
        wandb_params=WandBParams(use_wandb=False),
    )


def make_dataset(num_location_features: int = 5, num_attributes: int = 3, num_edge_features: int = 2):
    """Stand-in exposing only what ``build_feature_encoder`` and ``build_model`` read off a ``VisitDataset``."""
    schema = DatasetSchema(
        locations=continuous_schema("locations", num_location_features),
        edges=continuous_schema("edges", num_edge_features),
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=continuous_schema("user_attributes", num_attributes),
        distances=continuous_schema("distances", 1),
        targets=continuous_schema("targets", 1),
    )
    return types.SimpleNamespace(
        schema=schema,
        num_classes=2,
        edge_attr=torch.zeros(8, num_edge_features),
        num_rwpe_features=2,
    )


def record_training(monkeypatch, frame=None) -> list[dict]:
    """Replace model building and training with stubs, returning the list each training call is recorded into."""
    calls: list[dict] = []

    def train_and_evaluate_model(**kwargs):
        calls.append(kwargs)
        name = kwargs["variant"].name
        return frame(name) if frame else pl.DataFrame({"name": [name], "stage": ["test"]})

    monkeypatch.setattr(runners, "train_and_evaluate_model", train_and_evaluate_model)
    monkeypatch.setattr(
        runners, "build_feature_encoder", lambda variant, dataset, home_pe_cfg: types.SimpleNamespace(out_dim=1)
    )
    monkeypatch.setattr(runners, "build_model", lambda variant, dataset, in_channels: torch.nn.Linear(1, 1))

    return calls


def make_score_frame(name: str, seed: int | None = 42) -> pl.DataFrame:
    """Two score-vector rows, as ``score_vector_frame`` returns them."""
    return pl.DataFrame({
        "user_id": [7, 9],
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
        "user_id": [7, 9],
        "distances": [[0.0, 900.0, 4000.0], [4000.0, 900.0, 0.0]],
    }).with_columns(stage=pl.lit(HOME_DISTANCE_STAGE), epoch=pl.lit(None, dtype=pl.Int64))


def make_model_frame(name: str, seed: int | None = 42) -> pl.DataFrame:
    """One fit row, one test row, and two per-user rows, as ``train_and_evaluate_model`` returns them."""
    aggregate = pl.DataFrame([
        {"name": name, "stage": "fit", "epoch": 0, "train_loss": 0.5, "val_r_precision": 0.4},
        {"name": name, "stage": "test", "epoch": None, "test_r_precision": 0.48},
    ])
    per_user = pl.DataFrame({
        "user_id": [7, 9],
        "n_pos": [2, 3],
        "r_precision": [1.0, 0.5],
        "recall": [1.0, 0.33],
    }).with_columns(name=pl.lit(name), stage=pl.lit(PER_USER_STAGE), epoch=pl.lit(None, dtype=pl.Int64))

    combined = pl.concat([aggregate, per_user], how="diagonal")
    return combined.with_columns(seed=pl.lit(seed, dtype=pl.Int64))


class TestVariant:
    def test_name_appends_the_feature_flags_in_order(self, cfg):
        gat = cfg.models.gat

        assert Variant(gat).name == "GATv2-8"
        assert Variant(gat, full_info=True).name == "GATv2-8-dist"
        assert Variant(gat, use_home_pe=True).name == "GATv2-8-hpe"
        assert Variant(gat, full_info=True, use_home_pe=True).name == "GATv2-8-hpe-dist"

    def test_suffix_comes_last(self, cfg):
        assert Variant(cfg.models.mlp, full_info=True, suffix="nodemo").name == "MLP-dist-nodemo"

    def test_rwpe_free_variant_is_named_norwpe(self, cfg):
        assert Variant(cfg.models.mlp, full_info=True, use_rwpe=False).name == "MLP-dist-norwpe"

    def test_name_follows_an_overridden_depth(self, cfg):
        gat = with_overrides(cfg.models.gat, {"num_gcn_layers": 4})

        assert Variant(gat).name == "GATv2-4"
        assert cfg.models.gat.name == "GATv2-8"


class TestBuildModel:
    @pytest.mark.parametrize("key", ["mlp", "gat", "gps"])
    def test_every_architecture_builds(self, cfg, key):
        model = build_model(Variant(cfg.models[key]), make_dataset(), IN_CHANNELS)  # type: ignore[arg-type]

        assert isinstance(model, torch.nn.Module)

    def test_input_width_follows_the_variant_features(self, cfg):
        dataset = make_dataset(num_location_features=5, num_attributes=3)
        plain = build_feature_encoder(Variant(cfg.models.mlp, use_demographics=False), dataset, HOME_PE)  # type: ignore[arg-type]
        dist = build_feature_encoder(Variant(cfg.models.mlp, full_info=True), dataset, HOME_PE)  # type: ignore[arg-type]

        assert plain.out_dim == 5 + 1 + 2
        assert dist.out_dim == 5 + 1 + 2 + 3 + 1

    def test_rwpe_free_input_width_leaves_out_the_encoding(self, cfg):
        dataset = make_dataset(num_location_features=5, num_attributes=3)
        encoder = build_feature_encoder(Variant(cfg.models.mlp, full_info=True, use_rwpe=False), dataset, HOME_PE)  # type: ignore[arg-type]

        assert encoder.out_dim == 5 + 1 + 3 + 1

    def test_home_pe_adds_its_bins(self, cfg):
        dataset = make_dataset()
        dataset.home_hop_distance = torch.zeros(4, 4)
        plain = build_feature_encoder(Variant(cfg.models.mlp), dataset, HOME_PE)  # type: ignore[arg-type]
        with_pe = build_feature_encoder(Variant(cfg.models.mlp, use_home_pe=True), dataset, HOME_PE)  # type: ignore[arg-type]

        assert with_pe.out_dim - plain.out_dim == HOME_PE.bins

    def test_the_model_takes_the_given_input_width(self, cfg):
        model = build_model(Variant(cfg.models.mlp), make_dataset(), IN_CHANNELS)  # type: ignore[arg-type]

        assert isinstance(model, NodeMLP)
        assert model.lins[0].in_features == IN_CHANNELS

    def test_edge_dim_reaches_the_networks_that_take_it(self, cfg):
        dataset = make_dataset(num_edge_features=2)
        gat = build_model(Variant(cfg.models.gat), dataset, IN_CHANNELS)  # type: ignore[arg-type]
        gps = build_model(Variant(cfg.models.gps), dataset, IN_CHANNELS)  # type: ignore[arg-type]

        assert isinstance(gat, GAT) and isinstance(gps, GraphTransformer)
        assert gps.edge_proj.in_features == 2

    def test_gat_norm_follows_the_config(self, cfg):
        normed = build_model(Variant(cfg.models.gat), make_dataset(), IN_CHANNELS)  # type: ignore[arg-type]
        plain = build_model(Variant(with_overrides(cfg.models.gat, {"norm": None})), make_dataset(), IN_CHANNELS)  # type: ignore[arg-type]

        assert all(isinstance(norm, torch.nn.LayerNorm) for norm in normed.convs.norms)  # type: ignore[union-attr]
        assert all(isinstance(norm, torch.nn.Identity) for norm in plain.convs.norms)  # type: ignore[union-attr]

    def test_unknown_gat_norm_raises(self, cfg):
        with pytest.raises(KeyError, match="normalisation"):
            build_model(Variant(with_overrides(cfg.models.gat, {"norm": "nope"})), make_dataset(), IN_CHANNELS)  # type: ignore[arg-type]


class TestTrainerConfig:
    def test_every_key_is_a_trainer_argument(self, cfg):
        assert set(cfg.trainer) <= set(inspect.signature(L.Trainer.__init__).parameters)


class TestExperimentConfigs:
    @pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda path: path.stem)
    def test_every_experiment_config_names_a_runner(self, path):
        assert OmegaConf.load(path).experiment.name in EXPERIMENTS

    def test_comparison_is_the_default(self, cfg):
        assert cfg.experiment.name == "comparison"

    def test_overfit_health_sets_what_the_check_reads(self, compose_config):
        cfg = compose_config("experiment=overfit_health")

        assert cfg.experiment.name == "overfit_health"
        assert cfg.trainer.overfit_batches > 0
        assert cfg.train.log_train_ranking is True
        assert len(cfg.train.train_seeds) == 1

    def test_the_command_line_overrides_an_experiment_setting(self, compose_config):
        cfg = compose_config("experiment=depth_sweep", "experiment.depths=[3]")

        assert cfg.experiment.depths == [3]

    def test_overfit_health_rejects_a_run_that_does_not_overfit(self, compose_config, monkeypatch):
        cfg = compose_config("experiment=overfit_health", "trainer.overfit_batches=0")
        monkeypatch.setattr(runners, "setup_experiment", lambda cfg, **kw: pytest.fail("set up a run"))

        with pytest.raises(ValueError, match="overfit_batches"):
            runners.overfit_health_experiment(cfg)


class TestArchitectureRegistry:
    @pytest.mark.parametrize("path", sorted((get_project_root() / "activitygraphs/conf/model").glob("*.yaml")))
    def test_every_model_config_names_a_registered_architecture(self, path):
        assert OmegaConf.load(path).architecture in BUILDERS

    def test_an_unregistered_architecture_raises(self, cfg):
        model = with_overrides(cfg.models.mlp, {"architecture": "unknown"})

        with pytest.raises(KeyError, match="unknown"):
            architecture_builder(model)


class TestRunVariants:
    def _variants(self, cfg) -> list[Variant]:
        return [Variant(cfg.models.mlp), Variant(cfg.models.gat, full_info=True)]

    def test_trains_every_variant_once_per_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        frames = run_variants(make_setup(cfg, [1, 2]), self._variants(cfg))

        assert len(frames) == 4
        assert [call["variant"].name for call in calls] == ["MLP", "GATv2-8-dist", "MLP", "GATv2-8-dist"]

    def test_tags_each_frame_with_its_training_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        seeds = pl.concat(run_variants(make_setup(cfg, [1, 2]), self._variants(cfg)))["seed"].to_list()

        assert seeds == [1, 1, 2, 2]

    def test_builds_a_fresh_model_for_every_seed(self, cfg, monkeypatch):
        record_training(monkeypatch)
        built: list[torch.nn.Module] = []

        def build(variant, dataset, in_channels):
            built.append(torch.nn.Linear(1, 1))
            return built[-1]

        monkeypatch.setattr(runners, "build_model", build)
        run_variants(make_setup(cfg, [1, 2]), [Variant(cfg.models.mlp)])

        assert len(built) == 2
        assert built[0] is not built[1]

    def test_run_tag_separates_checkpoints_across_seeds(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        run_variants(make_setup(cfg, [1, 2]), self._variants(cfg))

        assert [call["run_tag"] for call in calls] == ["seed1", "seed1", "seed2", "seed2"]

    def test_no_run_tag_for_a_single_seed(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        run_variants(make_setup(cfg, [42]), self._variants(cfg))

        assert [call["run_tag"] for call in calls] == [None, None]

    def test_forwards_each_variant(self, cfg, monkeypatch):
        calls = record_training(monkeypatch)
        run_variants(make_setup(cfg, [42]), self._variants(cfg))

        assert [call["variant"] for call in calls] == self._variants(cfg)


class TestVariantHyperparams:
    def test_logs_the_architecture_and_flags(self, cfg):
        hyperparams = variant_hyperparams(Variant(cfg.models.gat, full_info=True))

        assert hyperparams["architecture"]["num_gcn_layers"] == cfg.models.gat.num_gcn_layers
        assert hyperparams["full_info"] is True


class TestDropEmptyColumns:
    def test_drops_all_null_columns_only(self):
        df = pl.DataFrame({"kept": [1, None], "dropped": [None, None]})

        assert drop_empty_columns(df).columns == ["kept"]

    def test_empty_frame_is_returned_unchanged(self):
        df = pl.DataFrame({"a": [], "b": []})

        assert drop_empty_columns(df).columns == ["a", "b"]


class TestDemographicsAblation:
    """The ablation must run both arms of `use_demographics` for each retained architecture."""

    def _recorded(self, cfg, monkeypatch) -> list[dict]:
        calls = record_training(monkeypatch, frame=make_model_frame)
        setup = make_setup(cfg, [42, 43])
        monkeypatch.setattr(runners, "setup_experiment", lambda cfg, **kw: setup)
        monkeypatch.setattr(runners, "save_results", lambda *a, **kw: None)

        run_cfg = types.SimpleNamespace(
            train=types.SimpleNamespace(),
            paths=types.SimpleNamespace(reports="reports"),
            data=types.SimpleNamespace(name="CMAP"),
        )
        runners.demographics_ablation_experiment(run_cfg)  # type: ignore[arg-type]

        return calls

    def test_runs_both_arms_of_each_architecture(self, cfg, monkeypatch):
        names = {call["variant"].name for call in self._recorded(cfg, monkeypatch)}

        assert names == {"MLP-dist-demo", "MLP-dist-nodemo", "GATv2-8-dist-demo", "GATv2-8-dist-nodemo"}

    def test_the_arms_differ_only_in_use_demographics(self, cfg, monkeypatch):
        variants = {call["variant"].name: call["variant"] for call in self._recorded(cfg, monkeypatch)}

        assert variants["MLP-dist-demo"].use_demographics is True
        assert variants["MLP-dist-nodemo"].use_demographics is False
        assert all(variant.full_info for variant in variants.values())

    def test_every_arm_runs_once_per_seed(self, cfg, monkeypatch):
        assert len(self._recorded(cfg, monkeypatch)) == 8  # 2 architectures x 2 arms x 2 seeds


class TestDepthSweep:
    """Depth survives only inside the model name unless it is logged, so grouping by it must be possible."""

    def _recorded(self, cfg, monkeypatch, depths: list[int]) -> list[dict]:
        calls = record_training(monkeypatch, frame=make_model_frame)
        setup = make_setup(cfg, [42])
        monkeypatch.setattr(runners, "setup_experiment", lambda cfg, **kw: setup)
        monkeypatch.setattr(runners, "save_results", lambda *a, **kw: None)

        run_cfg = types.SimpleNamespace(
            experiment=types.SimpleNamespace(depths=depths),
            paths=types.SimpleNamespace(reports="reports"),
            data=types.SimpleNamespace(name="CMAP"),
        )
        runners.depth_sweep_experiment(run_cfg)  # type: ignore[arg-type]

        return calls

    def test_every_run_logs_its_depth(self, cfg, monkeypatch):
        calls = self._recorded(cfg, monkeypatch, [2, 4, 8])

        assert [variant_hyperparams(call["variant"])["depth"] for call in calls] == [0, 2, 4, 8]

    def test_the_mlp_anchor_is_depth_zero(self, cfg, monkeypatch):
        anchor = self._recorded(cfg, monkeypatch, [2])[0]

        assert anchor["variant"].name == "MLP-dist"
        assert variant_hyperparams(anchor["variant"])["depth"] == 0

    def test_each_depth_is_named_after_it(self, cfg, monkeypatch):
        names = [call["variant"].name for call in self._recorded(cfg, monkeypatch, [2, 4])]

        assert names == ["MLP-dist", "GATv2-2-hpe-dist", "GATv2-4-hpe-dist"]


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

        assert "user_id" not in aggregate.columns
        assert "val_r_precision" not in per_user.columns
        assert {"name", "seed", "user_id", "n_pos", "r_precision", "recall"}.issubset(per_user.columns)

    def test_seed_is_retained_for_pairing(self, tmp_path):
        frames = [make_model_frame("MLP", seed=1), make_model_frame("MLP", seed=2)]
        _, per_user = self._saved(tmp_path, *frames)

        assert per_user["seed"].to_list() == [1, 1, 2, 2]
        assert per_user["user_id"].to_list() == [7, 9, 7, 9]

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
        assert {"name", "seed", "user_id"}.issubset(scores.columns)

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
