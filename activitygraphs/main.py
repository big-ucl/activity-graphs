"""Hydra entry points: ``main`` launches to the configured experiment, ``report_results`` analyses the results."""

import hydra
import torch

from activitygraphs.analysis import print_report
from activitygraphs.config import Config
from activitygraphs.experiments import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    """Run the experiment selected by the ``experiment`` config group."""
    torch.set_float32_matmul_precision("high")
    run_experiment(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="report")
def report_results(cfg: Config):
    """Print the results report for a recorded run."""
    print_report(
        cfg.paths.runs,
        cfg.data.name,
        cfg.train.max_recall_k,
        cfg.data.distance_bands,
        cfg.analysis.run,
        cfg.analysis,
    )


if __name__ == "__main__":
    main()
