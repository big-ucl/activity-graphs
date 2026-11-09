# Adding a model

A model in activity-graphs scores every node of the location graph for every user of a batch. The training loop, the
losses, and the metrics are shared by every model, and they live in `RankingModule`
(`activitygraphs/training/lightning_module.py`). A new model therefore only has to return logits of shape
`[num_users, num_nodes]`. It never needs its own training loop or metric code.

## Vocabulary

Four words name the parts of a model, and the code uses them consistently:

- A **network** is the inner `torch.nn.Module`, such as `NodeMLP`, `GAT`, or `GraphTransformer`
  (`activitygraphs/models/networks.py`).
- A **scorer** is the object held by `RankingModule`. It maps the locations and a `UserBatch` to `[B, N]` logits
  (`activitygraphs/models/scorers.py`).
- A **model** is one scorer plus its training settings. Its config is typed by `ModelConfig` and stored under
  `cfg.models.<key>`.
- A **variant** is a model paired with its input features, as trained by an experiment
  (`activitygraphs/experiments/variants.py`).

## Choose the kind of model

The steps depend on how the new model reads its inputs. Pick one of three kinds:

1. **A network on the shared input.** The network reads the input built by the `FeatureEncoder`: the location
   features, `is_home`, and the optional encodings and demographics. `NodeMLP`, `GAT`, and `GraphTransformer` are of
   this kind. Follow steps 1 to 6.
2. **A scorer with its own inputs.** The scorer reads the locations and the batch directly, as `TwoTowerScorer` and
   `HomeZoneMFBaseline` do. Follow steps 1 to 6, and replace step 5 with step 5b.
3. **A baseline fitted once.** The baseline is fitted in closed form on the training users, outside the training loop,
   as `Gravity` and `VisitFrequency` are. Follow step 7 only.

The guide uses a running example of the first kind: a network called `MyNet`, with config key `mynet`.

A new input, such as a location feature, a user attribute, or an input flag, is a change to the features rather than
a new model. The guide [Adding a feature](adding-a-feature.md) covers each of these.

## 1. Write the network

The network goes in `activitygraphs/models/networks.py`. Its `forward` takes the four arguments
`(x, edge_index, edge_attr, batch)` and returns one output per node, shape `[num_pairs, out_channels]`:

```python
class MyNet(torch.nn.Module):
    """Short description of the network."""

    def __init__(self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        ...

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """Return one output per node, shape ``(num_nodes, out_channels)``."""
        ...
```

The scorer decides what the four arguments hold. `NodeScorer` flattens the input to `[B * N, F]` and passes `None`
for the three graph arguments. `GraphScorer` expands the batch into `B` copies of the location graph and passes the
edges, the edge features, and the copy index of each node. A network that needs the graph must therefore run in a
`GraphScorer`.

The input width, the output width, and the edge width come from the dataset and the variant. Take them as constructor
arguments, and never hardcode them, because the number of location features differs between datasets.

## 2. Declare the config dataclass

The hyperparameters of the network are typed by a dataclass in `activitygraphs/config.py`. Subclass `NetworkConfig`,
which already holds `architecture`, `name`, `lr`, `weight_decay`, `hidden_channels`, and `dropout`, and document each
new field under `Attributes`:

```python
@dataclass
class MyNetConfig(NetworkConfig):
    """Config for the ``MyNet`` network.

    Attributes:
        num_layers: Number of layers.
    """

    num_layers: int
```

A model with no hidden layers or dropout, such as `HomeZoneMFConfig`, subclasses `ModelConfig` instead.

## 3. Write the model config

The yaml file `activitygraphs/conf/model/mynet.yaml` holds the values of every field of `MyNetConfig`, and nothing
else:

```yaml
architecture: mynet
name: MyNet-${.num_layers}
lr: 1e-3
weight_decay: 1e-4

num_layers: 3
hidden_channels: 128
dropout: 0.2
```

The `architecture` key names the builder of the scorer in step 5. The `name` key is the name reported in the results,
and it may interpolate a hyperparameter, as `GATv2-${.num_gcn_layers}` does. An experiment that sets that
hyperparameter then gets a name that follows the new value. A hyperparameter that an experiment sets and that `name`
does not interpolate needs a suffix, returned by the suffix function of the architecture in `OPTION_SUFFIXES`
(`activitygraphs/experiments/variants.py`), as the options of the two-tower model have.

## 4. Add the model to the root config

The root config `activitygraphs/conf/config.yaml` composes every model into `cfg.models`. Add one line to its
`defaults` list:

```yaml
defaults:
  ...
  - model@models.mynet: mynet
```

The key under `models` must equal the `architecture` value of the yaml. The config test composes each model file as
`model@models.<architecture>`, so a key that differs from the architecture fails that test.

Once the line is in place, the command line can override any hyperparameter, for example
`uv run activity-graphs models.mynet.num_layers=4`.

## 5. Register the builder

The registry `BUILDERS` in `activitygraphs/models/architectures.py` maps each `architecture` key to the function that
builds its scorer. For a network on the shared input, write a small builder that sizes the network from
`NetworkSizes`, and wrap it with `_encoded_scorer`:

```python
def _build_mynet(config: MyNetConfig, sizes: NetworkSizes) -> MyNet:
    """Instantiate a ``MyNet`` from ``config``, sized by ``sizes``."""
    return MyNet(
        num_layers=config.num_layers,
        in_channels=sizes.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=sizes.out_channels,
        dropout=config.dropout,
    )


BUILDERS: dict[str, ScorerBuilder] = {
    ...
    "mynet": _encoded_scorer(_build_mynet, GraphScorer),
}
```

`_encoded_scorer` builds the `FeatureEncoder` for the variant's input features, sets `in_channels` to the width of the
encoder, and sets `out_channels` and `edge_dim` from the dataset schema. Pass `NodeScorer` for a network that reads
each (user, node) pair alone, and `GraphScorer` for a network that passes messages over the graph. Put
`_build_mynet` with the other builders, below the public `build_scorer`, and above `BUILDERS`.

## 5b. Register a scorer with its own inputs

A scorer that reads the locations and the batch directly subclasses `Scorer`, and implements
`forward(locations, batch) -> [num_users, num_nodes]`. The fields of `UserBatch` (`activitygraphs/datasets/batch.py`)
are the per-user inputs: `home_node`, `is_home`, `home_distance` (scaled), `home_distance_m` (metres), `home_hops`,
`attributes`, and `attributes_cat`. The locations hold the shared inputs: `x`, `edge_index`, `edge_attr`,
`random_walk_pe`, and `distance_m`. A scorer that holds a network also overrides `compile_network`, so that
`train.compile` compiles that network.

The builder of such a scorer takes the full `ScorerBuilder` signature,
`(config, features, locations, schema, home_pe_cfg)`, and is registered in `BUILDERS` directly:

```python
BUILDERS: dict[str, ScorerBuilder] = {
    ...
    "my_scorer": _build_my_scorer,
}
```

The builder decides which `InputFeatures` flags the scorer supports. It raises a `ValueError` on a flag that the scorer
cannot honour, as `_build_home_zone_mf` does, so that a variant never reports a flag that had no effect.
`_build_two_tower` shows how a scorer maps each flag to one of its own inputs.

## 6. Test and check the model

Three tests read the model configs and the registry automatically:

- `test_model_config_matches_its_dataclass` in `test/test_config.py` checks that each yaml has exactly the fields of
  its dataclass, once `"mynet": MyNetConfig` is added to `MODEL_CONFIGS`.
- `test_every_model_config_names_a_registered_architecture` in `test/test_experiments.py` checks that each yaml names a
  key of `BUILDERS`.
- `test_models_never_imports_training` in `test/test_imports.py` checks that no module under `models/` imports from
  `training/`.

Add the new key to the two parametrised tests of `TestBuildScorer` in `test/test_experiments.py`:
`test_every_architecture_builds` and `test_each_architecture_runs_in_its_scorer`. Write unit tests for the network
itself on the stand-in graphs of `test/graphs.py` (`make_locations`, `make_users`, `join_users`), which need no data.

The model then needs a variant in an experiment before it can be trained. The guide
[Adding an experiment](adding-an-experiment.md) shows how to write one, for example the entry:

```yaml
    - { model: mynet }
```

Check that the model can memorise a single batch before reading any comparison. The `overfit_health` experiment
trains the variants listed in `activitygraphs/conf/experiment/overfit_health.yaml`, so add an entry for the new model
there, and run:

```bash
uv run activity-graphs experiment=overfit_health train.wandb=false
```

A healthy model drives `train_avg_recall@50` close to 1.0 on the memorised batch. A model that stays far below 1.0
has a capacity or an optimisation problem, and its results in a comparison mean nothing.

## 7. Add a fitted baseline

A baseline is fitted once on the training users, and it is evaluated beside the trained models of every experiment.
It does not go in `cfg.models`, and it has no yaml file.

1. Write a `FittedScorer` subclass in `activitygraphs/models/baselines/`. Its `fit(train, val, locations)` fits the
   parameters and returns `self`. Its `forward(locations, batch)` returns the `[num_users, num_nodes]` logits from the
   batch itself. A distance baseline reads the distance in metres from `batch.home_distance_m`. A popularity baseline
   uses the helpers of `models/baselines/popularity.py`, which leave each node's own residents out of its visit count.
2. Export the class from `activitygraphs/models/baselines/__init__.py`, in the import list and in `__all__`.
3. Add a `BaselineSpec` with the reported name to `_closed_form_baselines` in
   `activitygraphs/experiments/baselines.py`. The list is fitted in order, so a baseline that reads another fitted
   baseline goes after it, as `ConditionalGravity` follows `Gravity`.
4. Write unit tests on the pattern of `test/test_gravity.py` and `test/test_baselines.py`.

The baselines run in every experiment except `overfit_health`. Run them alone with:

```bash
uv run activity-graphs experiment=baselines
```
