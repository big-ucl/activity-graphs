"""Tests for the run directories: numbering, the latest finished run, and resolving a run number."""

from datetime import datetime

import pytest

from activitygraphs.runs import (
    RESULTS_FILE,
    create_overfit_dir,
    create_run_dir,
    latest_overfit_dir,
    latest_run,
    run_dir,
)

STARTED = datetime(2026, 10, 26, 15, 30, 0)


def make_run(runs_path, name: str, run: int, started: str = "20261026-153000", finished: bool = True):
    """A run directory, with a results file when ``finished``."""
    path = runs_path / f"{name}-{run}-{started}"
    path.mkdir(parents=True)
    if finished:
        (path / RESULTS_FILE).touch()

    return path


class TestCreateRunDir:
    def test_the_first_run_is_number_one(self, tmp_path):
        assert create_run_dir(tmp_path / "runs", "TPG", STARTED) == tmp_path / "runs" / "TPG-1-20261026-153000"

    def test_numbers_follow_the_highest_taken(self, tmp_path):
        make_run(tmp_path, "TPG", 1)
        make_run(tmp_path, "TPG", 7)

        assert create_run_dir(tmp_path, "TPG", STARTED).name == "TPG-8-20261026-153000"

    def test_a_run_without_results_still_takes_its_number(self, tmp_path):
        make_run(tmp_path, "TPG", 3, finished=False)

        assert create_run_dir(tmp_path, "TPG", STARTED).name == "TPG-4-20261026-153000"

    def test_other_datasets_do_not_take_numbers(self, tmp_path):
        make_run(tmp_path, "CMAP", 9)

        assert create_run_dir(tmp_path, "TPG", STARTED).name == "TPG-1-20261026-153000"


class TestLatestRun:
    def test_picks_the_highest_finished_number(self, tmp_path):
        for run in [1, 2, 10]:
            make_run(tmp_path, "TPG", run)
        make_run(tmp_path, "TPG", 11, finished=False)
        make_run(tmp_path, "OtherCity", 99)

        assert latest_run(tmp_path, "TPG") == 10

    def test_raises_when_no_run_has_results(self, tmp_path):
        make_run(tmp_path, "TPG", 1, finished=False)

        with pytest.raises(FileNotFoundError):
            latest_run(tmp_path, "TPG")


class TestRunDir:
    def test_resolves_a_run_number(self, tmp_path):
        first = make_run(tmp_path, "TPG", 1)
        make_run(tmp_path, "TPG", 2)

        assert run_dir(tmp_path, "TPG", 1) == first

    def test_none_resolves_the_latest_finished_run(self, tmp_path):
        make_run(tmp_path, "TPG", 1)
        second = make_run(tmp_path, "TPG", 2)
        make_run(tmp_path, "TPG", 3, finished=False)

        assert run_dir(tmp_path, "TPG") == second

    def test_a_missing_number_raises(self, tmp_path):
        make_run(tmp_path, "TPG", 1)

        with pytest.raises(FileNotFoundError):
            run_dir(tmp_path, "TPG", 2)

    def test_a_number_taken_twice_raises(self, tmp_path):
        make_run(tmp_path, "TPG", 4, started="20261026-153000")
        make_run(tmp_path, "TPG", 4, started="20261026-153001")

        with pytest.raises(ValueError, match="several"):
            run_dir(tmp_path, "TPG", 4)


class TestOverfitDir:
    def test_is_unnumbered(self, tmp_path):
        assert create_overfit_dir(tmp_path, "TPG", STARTED).name == "TPG-20261026-153000"

    def test_latest_is_the_most_recently_started(self, tmp_path):
        create_overfit_dir(tmp_path, "TPG", datetime(2026, 10, 25, 9, 0, 0))
        latest = create_overfit_dir(tmp_path, "TPG", STARTED)
        create_overfit_dir(tmp_path, "CMAP", datetime(2026, 10, 27, 9, 0, 0))

        assert latest_overfit_dir(tmp_path, "TPG") == latest
