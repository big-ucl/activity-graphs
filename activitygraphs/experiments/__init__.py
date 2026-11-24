"""[Stage 6: dataset -> trained models -> results] This package runs the experiment the ``experiment`` config group selects.

It is split into the following modules:
  - ``runners``: the per-seed training loop, and ``run_experiment``, which trains the variant list of the experiment
    and runs the restored variants and the baselines.
  - ``setup``: ``setup_experiment``, the split, loss, baselines, architectures and training settings an experiment's
    models share.
  - ``variants``: ``Variant``, an architecture from ``cfg.models`` with its input features, its name, and the
    expansion of an experiment's variant list.
  - ``restore``: restoration of variants of a recorded run from checkpoints, to be tested or fine-tuned in the current run.
  - ``baselines``: fit and evaluate every baseline.
  - ``training``: train and evaluate one model or baseline through Lightning, returning its result frames.
  - ``results``: write an experiment's result frames to the parquet files of the run directory.
"""

from activitygraphs.experiments.runners import run_experiment

__all__ = ["run_experiment"]
