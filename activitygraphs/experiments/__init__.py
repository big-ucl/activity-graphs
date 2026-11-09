"""[Stage 6: dataset -> trained models -> results] This package runs the experiment the ``experiment`` config group selects.

It is split into the following modules:
  - ``runners``: ``run_experiment``, which trains the variant list of the experiment beside the baselines, and the
    per-seed training loop.
  - ``setup``: ``setup_experiment``, the split, loss, baselines, architectures and training settings an experiment's
    models share.
  - ``variants``: ``Variant``, an architecture from ``cfg.models`` with its input features, its name, and the
    expansion of an experiment's variant list.
  - ``baselines``: fit and evaluate every baseline.
  - ``training``: train and evaluate one model or baseline through Lightning, returning its result frames.
  - ``results``: write an experiment's result frames to the numbered parquet files.
"""

from activitygraphs.experiments.runners import run_experiment

__all__ = ["run_experiment"]
