# Adding an experiment

An experiment in activity-graphs is a list of variants, trained once per training seed beside the baselines. The list
lives in one yaml file of `activitygraphs/conf/experiment/`, and one runner, `run_experiment`
(`activitygraphs/experiments/runners.py`), trains the list of every experiment. The shared setup draws the split,
builds the loss, and fits the baselines, so a new experiment only chooses which variants to train. It needs no Python.

The guide follows a running example: an experiment called `width_sweep`, which sweeps the hidden width of the GAT. Its
config is `activitygraphs/conf/experiment/width_sweep.yaml`.

## 1. Write the experiment config

The config file starts with `# @package _global_`, so it can set any key of the root config, not only the keys under
`experiment`. It sets the name of the experiment, which labels its W&B runs, and its list of variants:

```yaml
# @package _global_
# GAT hidden-width sweep, beside the MLP and the baselines.
experiment:
  name: width_sweep
  variants:
    - { model: mlp }
    # GAT hidden widths swept.
    - { model: gat, sweep: { hidden_channels: [ 32, 64, 128, 256 ] } }
```

Any other setting that the experiment needs goes in the same file. For example, `overfit_health.yaml` sets
`trainer.overfit_batches: 1`, `train.train_seeds: [ 42 ]`, and `train.log_train_ranking: true`, and `headline.yaml`
sets `models.gat.num_gcn_layers: 2`.

## 2. Write the variants

Each entry of `variants` is typed by `VariantConfig` (`activitygraphs/config.py`) and has four keys:

- `model`: the key of the architecture in `cfg.models`, such as `mlp`, `gat`, `gps`, `two_tower` or `home_zone_mf`.
- `set`: model config fields overridden for this variant, such as `{ bias: node }`.
- `features`: flags of `InputFeatures` overridden for this variant, such as `{ use_demographics: false }`. Every flag
  is on by default: the home-anchored PE, the distance from home, the demographics, and the random-walk PE.
- `sweep`: one model config field mapped to a list of values. The entry then gives one variant per value, with the
  fields of `set` applied to each one.

Only `model` is required. An unknown key raises an error, and so does an unknown flag of `features`.

The command line can override one entry of the list by its position, for example
`experiment.variants.1.sweep.hidden_channels=[64,128]`.

## 3. Check the variant names

The results, the checkpoint directories, and the W&B runs are all keyed by the variant name, so every variant of an
experiment needs a distinct name. The name is built by `Variant.name` (`activitygraphs/experiments/variants.py`) from
three parts, in this order:

1. the `name` of the model config, such as `GATv2-${.num_gcn_layers}`, which can interpolate a hyperparameter;
2. the option suffixes returned by the function of its architecture in `OPTION_SUFFIXES`, such as `nodebias` or `d128`;
3. a suffix for every flag of `InputFeatures` that differs from its default, such as `nohpe` or `nodemo`.

A variant with the defaults has the name of its model alone: `MLP`, `GATv2-8`, `TwoTower-4`. Each model therefore has
exactly one name in every experiment, and two runs of the same model pair by name.

A field set by `set` or `sweep` must change the name, or `build_variants` raises an error. The hidden width of the
example is not in the name of the GAT yet, so the example needs a suffix function for the GAT, registered in
`OPTION_SUFFIXES`. The function returns a suffix for each option that differs from its value in
`activitygraphs/conf/model/gat.yaml`, on the pattern of `_two_tower_suffixes`:

```python
def _gat_suffixes(model: ModelConfig) -> list[str]:
    gat = cast(GATConfig, model)
    return [] if gat.hidden_channels == 128 else [f"w{gat.hidden_channels}"]


OPTION_SUFFIXES: dict[str, Callable[[ModelConfig], list[str]]] = {"gat": _gat_suffixes, "two_tower": _two_tower_suffixes}
```

The variants of the example are then named `MLP`, `GATv2-8-w32`, `GATv2-8-w64`, `GATv2-8` and `GATv2-8-w256`.

The baselines are not variants. `setup_experiment` fits them once, beside the variants of every experiment. The trained
`HomeZoneMF` is the exception: it is a model, so an experiment that reports it lists `{ model: home_zone_mf }`.

## 4. Keep the comparison paired

The split is drawn once from `train.split_seed`, and `setup_experiment` never reseeds it. Every model in every
experiment of a dataset is therefore scored on the same test users, and the per-user rows of two runs can be compared
paired. Do not override `train.split_seed` in an experiment config: `_run_variants` reseeds the training alone, so the
spread across seeds measures training noise only.

An experiment that overfits (`trainer.overfit_batches > 0`) skips the baselines, and it logs the best
`train_avg_recall@K` of each variant on the memorised batch.

## 5. Write the tests

`test_every_experiment_trains_its_variants` in `test/test_experiments.py` builds the variants of every experiment
config and compares their names with `EXPECTED_VARIANTS`. Add the new experiment there, with its names in training
order:

```python
EXPECTED_VARIANTS = {
    ...
    "width_sweep": ["MLP", "GATv2-8-w32", "GATv2-8-w64", "GATv2-8", "GATv2-8-w256"],
}
```

`test_experiment_config_matches_its_dataclass` in `test/test_config.py` checks that the `experiment` node has exactly
the fields of `ExperimentConfig`. A new suffix needs a test in `TestVariant` of `test/test_experiments.py`.
`test_the_models_as_configured_have_no_suffixes` checks that the reference values of a suffix function match the yaml.

## 6. Run the experiment

Start with a single-batch run, which runs one batch of every variant and writes no run directory:

```bash
uv run activity-graphs experiment=width_sweep trainer.fast_dev_run=true train.wandb=false
```

Then run the experiment and read its report:

```bash
uv run activity-graphs experiment=width_sweep data=cmap
uv run report-results data=cmap
```

Each run writes a numbered directory `reports/runs/{dataset}-{N}-{timestamp}/`, with the config used by the run, the
result parquet files, and the checkpoints of every variant. `report-results` reads the latest run of the dataset by
default, and `analysis.run=N` picks another one. Read the paired comparison against `analysis.reference_model`
(`Gravity` by default) rather than the aggregate means, and report each metric as mean +- sd over the training seeds.

## 7. Update the documentation

The README lists every experiment under "`experiment=<name>` selects what to run". Add a bullet for the new
experiment, with one sentence on what it trains.
