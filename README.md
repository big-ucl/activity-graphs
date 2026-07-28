# Activity Graphs

A graph neural network framework for predicting location consideration sets from travel surveys. Given a survey participant's home location and demographics, the model scores every zone of a city's spatial network and ranks them into a tractable candidate set of locations the person plausibly considers visiting. This choice set can be used as input to downstream activity-based travel models. The task is set up as node-level classification/ranking over a spatial adjacency graph of census/traffic analysis zones, trained on realised visits. 

Datasets: Toronto (THATS), Geneva (TPG), with Chicago (CMAP MyDailyTravel) coming soon. Survey data is not distributed in this repository, `data/` must be added to the local repo clone.

## Setup

Requires [uv](https://docs.astral.sh/uv/). PyTorch / PyTorch Geometric are pulled from CUDA 12.6 wheel indices configured in `pyproject.toml`.

From the cloned repository, run:
```bash
uv sync
```
to setup the project and install dependencies.

## Usage

```bash
uv run activity-graphs                       # full comparison run, default dataset (Toronto)
uv run activity-graphs data=geneva           # switch dataset (geneva, toronto)

uv run activity-graphs train.fast_dev_run=true      # single batch test run
uv run activity-graphs train.loss.type=bce          # loss: bpr (default) or bce
uv run activity-graphs train.loss.neg_sampler=hard  # hard near-home BPR negatives
uv run activity-graphs train.wandb=false            # disable Weights & Biases logging
```

Configuration is Hydra-based (`activitygraphs/conf/`). Any config key can be overridden on the command line. A run trains the GNN variants (GATSkip, GraphTransformer, MLP) against frequency baselines and writes per-epoch metrics to WandB and `reports/data/{name}-results-{N}.parquet`, with best-validation checkpoints in `models/`.

## Tests

```bash
uv run pytest test/                    # unit tests (no data required)
uv run pytest test/ -m integration     # integration tests (skip when data is absent)
```

## Layout

```
activitygraphs/
  main.py, run.py      # Hydra entry point and experiment running
  network.py           # Abstraction for travel data (NetworkData)
  dataprocessing.py    # Graph construction and pytorch tensor building
  data/                # Per-dataset loaders (geneva, toronto, ...)
  conf/                # Hydra configs (root + per-dataset)
  ml/                  # PyG dataset, Lightning module, models, losses, metrics, baselines
test/                  # pytest suite
reports/data/          # result dataframes
```

Models are evaluated using metrics for consideration sets: realised visits are a subset of the latent target, so we use recall metrics that are set size-aware (R-precision, recall at a set-size budget). Precision-style metrics are diagnostics only.
