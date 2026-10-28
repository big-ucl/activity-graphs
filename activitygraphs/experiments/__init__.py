"""[Stage 6: dataset -> trained models -> results] This package runs the experiments ``experiment.name`` selects.

It is split into the following modules:
  - ``runners``: one function per experiment, each a list of variants trained beside the baselines, and the
    per-seed training loop.
  - ``setup``: ``setup_experiment``, the split, loss, baselines, architectures and training settings an experiment's
    models share.
  - ``variants``: ``Variant``, an architecture from ``cfg.models`` with its input features, and ``build_model``.
  - ``architectures``: ``BUILDERS``, the registry mapping a model config's ``architecture`` key to the builder of
    its model.
  - ``baselines``: fit and evaluate every baseline.
  - ``training``: train and evaluate one model or baseline through Lightning, returning its result frames.
  - ``results``: write an experiment's result frames to the numbered parquet files.
"""

from activitygraphs.experiments.runners import (
    baselines_experiment,
    comparison_experiment,
    core_experiment,
    demographics_ablation_experiment,
    depth_sweep_experiment,
    overfit_health_experiment,
)

__all__ = [
    "baselines_experiment",
    "comparison_experiment",
    "core_experiment",
    "demographics_ablation_experiment",
    "depth_sweep_experiment",
    "overfit_health_experiment",
]
