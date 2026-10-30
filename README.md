# Activity Graphs

A graph neural network framework for predicting location consideration sets from travel surveys. Given a survey
participant's home location and demographics, the model scores every zone of a city's spatial network and ranks them
into a tractable candidate set of locations the person plausibly considers visiting. This choice set can be used as
input to downstream activity-based travel models. The task is set up as node-level classification/ranking over a spatial
adjacency graph of census/traffic analysis zones, trained on realised visits.

Datasets: Geneva (TPG) and Chicago (CMAP MyDailyTravel), with Toronto (TTS) coming soon. The Toronto THATS loader is
kept but no longer used. Survey data is not distributed in this repository, `data/` must be added to the local repo
clone.

## Setup

Requires [uv](https://docs.astral.sh/uv/). PyTorch / PyTorch Geometric are pulled from CUDA 12.6 wheel indices
configured in `pyproject.toml`.

From the cloned repository, run:

```bash
uv sync
```

to setup the project and install dependencies.

## Usage

```bash
uv run activity-graphs                       # full comparison run, default dataset (TPG)
uv run activity-graphs data=cmap             # switch dataset (tpg, cmap, thats)

uv run activity-graphs trainer.fast_dev_run=true    # single batch test run
uv run activity-graphs train.loss.type=bce          # loss: bpr (default) or bce
uv run activity-graphs train.loss.neg_sampler=hard  # hard near-home BPR negatives
uv run activity-graphs train.wandb=false            # disable Weights & Biases logging

uv run activity-graphs train.train_seeds=[1,2,3]    # repeat every model for each training seed
uv run activity-graphs experiment=core              # the short run for paired comparisons between code changes
uv run activity-graphs experiment=depth_sweep       # sweep GAT depth instead of comparing architectures
uv run activity-graphs experiment=depth_sweep experiment.depths=[2,4,8]
```

Once a run has finished, report its results:

```bash
uv run report-results                                     # analyse the latest run of the default dataset
uv run report-results data=tpg analysis.run=3             # a specific dataset and run number
uv run report-results analysis.reference_model=MLP-dist   # change reference model for comparisons
```

`report-results` reads the result Parquet files and prints:

- the aggregate metric as mean +- sd over the training seeds;
- the per-user paired comparison of every model against `analysis.reference_model` (bootstrap CI and Wilcoxon p);
- the recall@k curve, and the paired comparison split by whether the user's home zone was seen in training;
- the per-hop-band and per-distance-band metrics (AUC within each distance band); and
- the model health checks.

Configuration is Hydra-based (`activitygraphs/conf/`). Any config key can be overridden on the command line.

Feature transformations are done as follows:

1. a schema is declared for each tensor where every feature column is assigned a ``kind`` (continuous, count, binary or
   categorical)
2. The transforms (scaling, log1p, etc...) depend on the ``kind`` and are fitted on the training split. Continuous
   columns are z-scored, counts (the distances) are log1p then z-scored, and binary columns pass through. Categorical
   columns, such as the demographics, are embedded by the model.

`experiment=<name>` (a file of `activitygraphs/conf/experiment/`) selects what to run:

- `comparison` (default) trains the MLP, GATv2 and GraphTransformer variants against the baselines (popularity and
  distance decay),
- `core` trains five MLP, GATv2 and GraphTransformer variants. A shorter run for comparisons between code changes,
- `depth_sweep` sweeps the message-passing depth of the GAT variant,
- `demographics_ablation` trains the two best variants with and without the demographics,
- `baselines` fits the baselines alone,
- `overfit_health` checks that each architecture can memorise a single batch.

Each experiment writes per-epoch metrics to WandB and to a run directory `reports/runs/{name}-{N}-{timestamp}/`
(`reports/overfit/{name}-{timestamp}/` for overfit runs): the config
used by the run, `results.parquet`, per-test-user scores in `per-user.parquet`, the per-node score vectors in
`scores.parquet`, the per-node distances from home in `distances.parquet`, and the best and last checkpoints of
each model under `checkpoints/`. The train/val/test split is drawn with
`train.split_seed`; every model in every run is scored on the same test users;
`train.train_seeds` reseeds each model run to measure spread.

## Tests

```bash
uv run pytest test/                    # unit tests (no data required)
uv run pytest test/ -m integration     # integration tests (skip when data is absent)
```

## Layout

```
activitygraphs/
  main.py                  # Hydra entry points
  experiments/             # experiment runners, variants, architecture builders, training and baselines
  analysis.py              # Analysis of result Parquet files (seed spread, pairing, hop bands)
  schema.py                # column kinds and the tensor schemas
  surveys/                 # Abstraction for travel data (SurveyData), per-survey loaders (tpg, thats, cmap) and registry
  graph/                   # Graph construction
  tensors/                 # pytorch tensor building
  datasets/                # PyG dataset, splits, feature transforms, Lightning data module
  conf/                    # Hydra configs (root, per-dataset, per-model, per-experiment)
  ml/                      # Feature encoder, Lightning module, models, losses, metrics, baselines
test/                      # pytest suite
reports/runs/              # one directory per run: config, result dataframes, checkpoints, logs
```

Models are evaluated using metrics for consideration sets: realised visits are a subset of the latent target, so we use
recall metrics at a set-size budget. The headline is `avg_recall@50`, the mean recall@k over k = 1..50, read alongside
the recall@k curve. R-precision and precision-style metrics are diagnostics only. Each user's own home zone is left out
of the ranking.
