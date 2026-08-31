"""Tests for the seed loop and the aggregate / per-user result split."""

import types

import polars as pl
import torch

from activitygraphs import experiments

from activitygraphs.experiments import (
    ExperimentSetup,
    ModelSpec,
    drop_empty_columns,
    run_model_specs,
    save_results,
)
from activitygraphs.ml.training import PER_USER_STAGE, SCORE_VECTOR_STAGE, WandBParams


def make_setup(train_seeds: list[int], run_model=None) -> ExperimentSetup:
    """An ExperimentSetup with no datamodule; spec factories stay unevaluated so no data is needed."""
    return ExperimentSetup(
        cfg=None,  # type: ignore[arg-type]
        train_dataset=None,  # type: ignore[arg-type]
        run_model=run_model,  # type: ignore[arg-type]
        baseline_results=[],
        train_seeds=train_seeds,
        hidden_channels=128,
        dropout=0.2,
        lr_for={"MLP": 1e-3, "GATSkip": 1e-3}.__getitem__,
    )


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


class TestRunModelSpecs:
    def _specs(self) -> list[ModelSpec]:
        return [
            ModelSpec("MLP", lambda: torch.nn.Linear(1, 1), 1e-3, {}),
            ModelSpec("GATSkip-2-res", lambda: torch.nn.Linear(1, 1), 1e-3, {"full_info": True}),
        ]

    def _recording_setup(self, train_seeds):
        calls: list[dict] = []

        def run_model(model, name, lr, run_tag, **kwargs):
            calls.append({"name": name, "run_tag": run_tag, **kwargs})
            return pl.DataFrame({"name": [name], "stage": ["test"]})

        return make_setup(train_seeds, run_model), calls

    def test_trains_every_spec_once_per_seed(self):
        setup, calls = self._recording_setup([1, 2])
        frames = run_model_specs(setup, self._specs())

        assert len(frames) == 4
        assert [call["name"] for call in calls] == ["MLP", "GATSkip-2-res", "MLP", "GATSkip-2-res"]

    def test_tags_each_frame_with_its_training_seed(self):
        setup, _ = self._recording_setup([1, 2])
        seeds = pl.concat(run_model_specs(setup, self._specs()))["seed"].to_list()

        assert seeds == [1, 1, 2, 2]

    def test_builds_a_fresh_model_for_every_seed(self):
        setup, _ = self._recording_setup([1, 2])
        built: list[torch.nn.Module] = []

        def build():
            built.append(torch.nn.Linear(1, 1))
            return built[-1]

        run_model_specs(setup, [ModelSpec("MLP", build, 1e-3, {})])

        assert len(built) == 2
        assert built[0] is not built[1]

    def test_run_tag_separates_checkpoints_across_seeds(self):
        setup, calls = self._recording_setup([1, 2])
        run_model_specs(setup, self._specs())

        assert [call["run_tag"] for call in calls] == ["seed1", "seed1", "seed2", "seed2"]

    def test_no_run_tag_for_a_single_seed(self):
        setup, calls = self._recording_setup([42])
        run_model_specs(setup, self._specs())

        assert [call["run_tag"] for call in calls] == [None, None]

    def test_forwards_the_spec_train_and_eval_kwargs(self):
        setup, calls = self._recording_setup([42])
        run_model_specs(setup, self._specs())

        assert calls[1]["full_info"] is True


class TestDropEmptyColumns:
    def test_drops_all_null_columns_only(self):
        df = pl.DataFrame({"kept": [1, None], "dropped": [None, None]})

        assert drop_empty_columns(df).columns == ["kept"]

    def test_empty_frame_is_returned_unchanged(self):
        df = pl.DataFrame({"a": [], "b": []})

        assert drop_empty_columns(df).columns == ["a", "b"]


class TestDemographicsAblation:
    """The ablation must run both arms of `use_demographics` for each retained architecture."""

    def _recorded(self, monkeypatch) -> list[dict]:
        calls: list[dict] = []

        def run_model(model, name, lr, run_tag, **kwargs):
            calls.append({"name": name, **kwargs})
            return make_model_frame(name)

        setup = make_setup([42, 43], run_model=run_model)
        monkeypatch.setattr(experiments, "setup_experiment", lambda cfg, **kw: setup)
        monkeypatch.setattr(experiments, "build_mlp", lambda *a, **kw: torch.nn.Linear(1, 1))
        monkeypatch.setattr(experiments, "build_gat", lambda *a, **kw: torch.nn.Linear(1, 1))
        monkeypatch.setattr(experiments, "save_results", lambda *a, **kw: None)

        cfg = types.SimpleNamespace(
            train=types.SimpleNamespace(fast_dev_run=False),
            paths=types.SimpleNamespace(reports="reports"),
            data=types.SimpleNamespace(name="CMAP"),
        )
        experiments.demographics_ablation_experiment(cfg)

        return calls

    def test_runs_both_arms_of_each_architecture(self, monkeypatch):
        names = {call["name"] for call in self._recorded(monkeypatch)}

        assert names == {"MLP-dist-demo", "MLP-dist-nodemo", "GATSkip-8-res-dist-demo", "GATSkip-8-res-dist-nodemo"}

    def test_the_arms_differ_only_in_use_demographics(self, monkeypatch):
        calls = {call["name"]: call for call in self._recorded(monkeypatch)}

        assert calls["MLP-dist-demo"]["use_demographics"] is True
        assert calls["MLP-dist-nodemo"]["use_demographics"] is False
        assert all(call["full_info"] for call in calls.values())

    def test_every_arm_runs_once_per_seed(self, monkeypatch):
        assert len(self._recorded(monkeypatch)) == 8  # 2 architectures x 2 arms x 2 seeds


class TestDepthSweep:
    """Depth survives only inside the model name unless it is logged, so grouping by it must be possible."""

    def _recorded(self, monkeypatch, depths: list[int]) -> list[dict]:
        calls: list[dict] = []

        def run_model(model, name, lr, run_tag, **kwargs):
            calls.append({"name": name, **kwargs})
            return make_model_frame(name)

        setup = make_setup([42], run_model=run_model)
        monkeypatch.setattr(experiments, "setup_experiment", lambda cfg, **kw: setup)
        monkeypatch.setattr(experiments, "build_mlp", lambda *a, **kw: torch.nn.Linear(1, 1))
        monkeypatch.setattr(experiments, "build_gat", lambda *a, **kw: torch.nn.Linear(1, 1))
        monkeypatch.setattr(experiments, "save_results", lambda *a, **kw: None)

        cfg = types.SimpleNamespace(
            train=types.SimpleNamespace(fast_dev_run=False, depths=depths),
            paths=types.SimpleNamespace(reports="reports"),
            data=types.SimpleNamespace(name="CMAP"),
        )
        experiments.depth_sweep_experiment(cfg)

        return calls

    def test_every_run_logs_its_depth(self, monkeypatch):
        calls = self._recorded(monkeypatch, [2, 4, 8])

        assert [call["extra_hyperparams"]["depth"] for call in calls] == [0, 2, 4, 8]

    def test_the_mlp_anchor_is_depth_zero(self, monkeypatch):
        anchor = self._recorded(monkeypatch, [2])[0]

        assert anchor["name"] == "MLP-dist"
        assert anchor["extra_hyperparams"] == {"depth": 0}


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
    def _saved(self, tmp_path, *frames: pl.DataFrame, name: str = "TestSet"):
        (tmp_path / "data").mkdir()
        save_results(tmp_path, name, *frames)
        data = tmp_path / "data"
        return pl.read_parquet(data / f"{name}-results-1.parquet"), pl.read_parquet(data / f"{name}-per-user-1.parquet")

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

    def test_both_files_share_a_run_number(self, tmp_path):
        (tmp_path / "data").mkdir()
        save_results(tmp_path, "TestSet", make_model_frame("MLP"))
        save_results(tmp_path, "TestSet", make_model_frame("MLP"))

        written = sorted(f.name for f in (tmp_path / "data").iterdir())
        assert written == [
            "TestSet-per-user-1.parquet",
            "TestSet-per-user-2.parquet",
            "TestSet-results-1.parquet",
            "TestSet-results-2.parquet",
        ]

    def test_writes_only_the_aggregate_file_when_no_per_user_rows(self, tmp_path):
        (tmp_path / "data").mkdir()
        aggregate_only = make_model_frame("MLP").filter(pl.col("stage") != PER_USER_STAGE)
        save_results(tmp_path, "TestSet", aggregate_only)

        assert [f.name for f in (tmp_path / "data").iterdir()] == ["TestSet-results-1.parquet"]

    def test_no_scores_file_when_the_run_saved_no_score_vectors(self, tmp_path):
        (tmp_path / "data").mkdir()
        save_results(tmp_path, "TestSet", make_model_frame("MLP"))

        written = sorted(f.name for f in (tmp_path / "data").iterdir())
        assert written == ["TestSet-per-user-1.parquet", "TestSet-results-1.parquet"]

    def test_score_vectors_go_to_their_own_file(self, tmp_path):
        (tmp_path / "data").mkdir()
        combined = pl.concat([make_model_frame("MLP"), make_score_frame("MLP")], how="diagonal")
        save_results(tmp_path, "TestSet", combined)

        data = tmp_path / "data"
        assert sorted(f.name for f in data.iterdir()) == [
            "TestSet-per-user-1.parquet",
            "TestSet-results-1.parquet",
            "TestSet-scores-1.parquet",
        ]

        scores = pl.read_parquet(data / "TestSet-scores-1.parquet")
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

    def test_all_three_files_share_a_run_number(self, tmp_path):
        (tmp_path / "data").mkdir()
        combined = pl.concat([make_model_frame("MLP"), make_score_frame("MLP")], how="diagonal")
        save_results(tmp_path, "TestSet", combined)
        save_results(tmp_path, "TestSet", combined)

        written = sorted(f.name for f in (tmp_path / "data").iterdir())
        assert written == [
            "TestSet-per-user-1.parquet",
            "TestSet-per-user-2.parquet",
            "TestSet-results-1.parquet",
            "TestSet-results-2.parquet",
            "TestSet-scores-1.parquet",
            "TestSet-scores-2.parquet",
        ]
