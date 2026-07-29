"""Tests for the seed loop and the aggregate / per-user result split."""

import polars as pl
import torch

from activitygraphs.experiments import (
    ExperimentSetup,
    ModelSpec,
    drop_empty_columns,
    run_model_specs,
    save_results,
)
from activitygraphs.ml.training import PER_USER_STAGE


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
