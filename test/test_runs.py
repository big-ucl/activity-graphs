"""Tests for the run directories: the run ids, the latest finished run, and resolving a run id."""

from datetime import datetime

import pytest

from activitygraphs.runs import (
    RESULTS_FILE,
    _latest_run,
    create_run_dir,
    dataset_of_run,
    latest_overfit_dir,
    run_dir,
    run_id,
)

STARTED = datetime(2026, 10, 26, 15, 30, 0)


def make_run(root, run: str, finished: bool = True):
    """A run directory, with a results file when ``finished``."""
    path = create_run_dir(root, run)
    if finished:
        (path / RESULTS_FILE).touch()

    return path


class TestRunId:
    def test_joins_the_dataset_the_experiment_and_the_start(self):
        assert run_id("TTS2016", "edge_ablation", STARTED) == "TTS2016-edge_ablation-20261026-153000"

    def test_an_experiment_name_with_a_hyphen_raises(self):
        with pytest.raises(ValueError, match="not a run id"):
            run_id("TPG", "edge-ablation", STARTED)

    def test_the_dataset_is_read_back(self):
        assert dataset_of_run("TTS2016-edge_ablation-20261026-153000") == "TTS2016"

    def test_reading_the_dataset_of_a_run_number_raises(self):
        with pytest.raises(ValueError, match="not a run id"):
            dataset_of_run("21")


class TestCreateRunDir:
    def test_creates_the_directory_named_by_the_id(self, tmp_path):
        run = "TPG-core-20261026-153000"

        assert create_run_dir(tmp_path / "runs", run) == tmp_path / "runs" / run

    def test_an_id_taken_twice_raises(self, tmp_path):
        create_run_dir(tmp_path, "TPG-core-20261026-153000")

        with pytest.raises(FileExistsError):
            create_run_dir(tmp_path, "TPG-core-20261026-153000")


class TestLatestRun:
    def test_picks_the_latest_started_finished_run(self, tmp_path):
        make_run(tmp_path, "TPG-core-20261024-090000")
        make_run(tmp_path, "TPG-headline-20261025-090000")
        make_run(tmp_path, "TPG-core-20261026-090000", finished=False)
        make_run(tmp_path, "CMAP-core-20261027-090000")

        assert _latest_run(tmp_path, "TPG") == "TPG-headline-20261025-090000"

    def test_orders_by_the_start_not_by_the_experiment_name(self, tmp_path):
        make_run(tmp_path, "TPG-zzz-20261024-090000")
        make_run(tmp_path, "TPG-aaa-20261025-090000")

        assert _latest_run(tmp_path, "TPG") == "TPG-aaa-20261025-090000"

    def test_raises_when_no_run_has_results(self, tmp_path):
        make_run(tmp_path, "TPG-core-20261024-090000", finished=False)

        with pytest.raises(FileNotFoundError):
            _latest_run(tmp_path, "TPG")


class TestRunDir:
    def test_resolves_a_run_id(self, tmp_path):
        first = make_run(tmp_path, "TPG-core-20261024-090000")
        make_run(tmp_path, "TPG-core-20261025-090000")

        assert run_dir(tmp_path, "TPG", "TPG-core-20261024-090000") == first

    def test_none_resolves_the_latest_finished_run(self, tmp_path):
        make_run(tmp_path, "TPG-core-20261024-090000")
        second = make_run(tmp_path, "TPG-core-20261025-090000")
        make_run(tmp_path, "TPG-core-20261026-090000", finished=False)

        assert run_dir(tmp_path, "TPG") == second

    def test_a_missing_id_raises(self, tmp_path):
        make_run(tmp_path, "TPG-core-20261024-090000")

        with pytest.raises(FileNotFoundError):
            run_dir(tmp_path, "TPG", "TPG-core-20261025-090000")

    def test_a_run_of_another_dataset_raises(self, tmp_path):
        make_run(tmp_path, "CMAP-core-20261024-090000")

        with pytest.raises(ValueError, match="not a run of TPG"):
            run_dir(tmp_path, "TPG", "CMAP-core-20261024-090000")


class TestOverfitDir:
    def test_latest_is_the_most_recently_started(self, tmp_path):
        create_run_dir(tmp_path, "TPG-overfit_health-20261025-090000")
        latest = create_run_dir(tmp_path, "TPG-overfit_health-20261026-153000")
        create_run_dir(tmp_path, "CMAP-overfit_health-20261027-090000")

        assert latest_overfit_dir(tmp_path, "TPG") == latest
