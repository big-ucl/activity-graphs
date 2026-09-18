"""Hydra entry points: ``main`` launches to the configured experiment, ``report_results`` analyses the results."""

import hydra
from hydra.core.config_store import ConfigStore

from activitygraphs.analysis import print_report
from activitygraphs.config import Config
from activitygraphs.experiments import (
    baselines_experiment,
    comparison_experiment,
    demographics_ablation_experiment,
    depth_sweep_experiment,
    overfit_health_experiment,
)

cs = ConfigStore.instance()
cs.store(name="geneva_config", node=Config)

EXPERIMENTS = {
    "comparison": comparison_experiment,
    "depth_sweep": depth_sweep_experiment,
    "demographics_ablation": demographics_ablation_experiment,
    "overfit_health": overfit_health_experiment,
    "baselines": baselines_experiment,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    if cfg.train.experiment not in EXPERIMENTS:
        raise ValueError(f"train.experiment must be one of {sorted(EXPERIMENTS)}, got {cfg.train.experiment}")

    experiment = EXPERIMENTS[cfg.train.experiment]
    experiment(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="report")
def report_results(cfg: Config):
    print_report(cfg.paths.reports_data, cfg.data.name, cfg.train.max_recall_k, cfg.analysis.run, cfg.analysis)


if __name__ == "__main__":
    main()
