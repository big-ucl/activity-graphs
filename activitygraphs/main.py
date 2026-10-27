"""Hydra entry points: ``main`` launches to the configured experiment, ``report_results`` analyses the results."""

import hydra

from activitygraphs.analysis import print_report
from activitygraphs.config import Config
from activitygraphs.experiments import (
    baselines_experiment,
    comparison_experiment,
    demographics_ablation_experiment,
    depth_sweep_experiment,
    overfit_health_experiment,
    rwpe_control_experiment,
)

EXPERIMENTS = {
    "comparison": comparison_experiment,
    "depth_sweep": depth_sweep_experiment,
    "demographics_ablation": demographics_ablation_experiment,
    "rwpe_control": rwpe_control_experiment,
    "overfit_health": overfit_health_experiment,
    "baselines": baselines_experiment,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    """Run the experiment selected by ``experiment.name``."""
    if cfg.experiment.name not in EXPERIMENTS:
        raise ValueError(f"experiment.name must be one of {sorted(EXPERIMENTS)}, got {cfg.experiment.name}")

    experiment = EXPERIMENTS[cfg.experiment.name]
    experiment(cfg)


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
