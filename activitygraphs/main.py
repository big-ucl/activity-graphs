"""Hydra entry point: registers config store and dispatches to the configured experiment."""

import hydra
from hydra.core.config_store import ConfigStore

from activitygraphs.config import Config
from activitygraphs.experiments import comparison_experiment, depth_sweep_experiment

cs = ConfigStore.instance()
cs.store(name="geneva_config", node=Config)

EXPERIMENTS = {"comparison": comparison_experiment, "depth_sweep": depth_sweep_experiment}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    if cfg.train.experiment not in EXPERIMENTS:
        raise ValueError(f"train.experiment must be one of {sorted(EXPERIMENTS)}, got {cfg.train.experiment}")

    experiment = EXPERIMENTS[cfg.train.experiment]
    experiment(cfg)


if __name__ == "__main__":
    main()
