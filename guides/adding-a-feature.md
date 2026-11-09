# Adding a feature

A feature reaches a model through four stages. The survey and the graph builder compute the raw values. The tensor
builder stores them with a schema that gives each column a kind. The feature transforms scale each column by its kind,
fitted on the training users. Finally, the `FeatureEncoder` of each variant concatenates its input sources, one source
per input flag. A new feature enters at the stage that matches what it describes.

## Choose the kind of feature

The right place for a feature depends on what the feature has one value for. There are four cases:

| The feature has one value per... | Example | Where it is computed | Model code needed |
|---|---|---|---|
| zone | population density, POI counts | the graph builder | none |
| graph edge | centroid distance (`weight`) | the graph builder | none |
| user | age band, vehicle count | the survey module | none |
| (user, zone) pair, or a graph encoding | distance from home, home PE, RWPE | `UserBatch.join` or `Locations.from_tensors` | a `FeatureSource` and a flag |

The first three cases change the input of every variant that reads the group. A location feature reaches every
variant, a user attribute reaches every variant with `use_demographics`, and an edge feature reaches every network
that passes messages. The fourth case adds an input flag, so only the variants that set the flag see the feature.

The model conditions on the home node alone. A feature of a (user, zone) pair must therefore be derived from the
user's home node and the shared graph. It must never be derived from the user's other trip ends, such as a work or a
study location, because those trip ends are the visits that the model predicts.

## Column kinds and transforms

Every feature column has a kind, declared in its `TensorSchema` (`activitygraphs/schema.py`). The kind decides the
transform that `TRANSFORM_POLICY` (`activitygraphs/datasets/transforms.py`) applies to the column:

| Kind | Transform | Typical columns |
|---|---|---|
| `continuous` | z-score | densities, shares of area, adult and child counts |
| `count` | log1p, then z-score | distances, edge weights |
| `binary` | none | `is_home`, 0/1 indicators |
| `categorical` | none, embedded by the model | the demographic categories |

The transforms are fitted on the training split only. The location and edge features are fitted over every node and
edge, since the graph is shared by every user. The user attributes are fitted over the training users, and the
distances over the rows of the training users' home nodes. A column that is constant on the fitted rows is z-scored to
0, and the fit logs a warning for it.

A categorical column is stored as integer codes, and `SchemaEncoder` (`activitygraphs/models/schema_encoder.py`) gives
it a learned embedding. `categories[i]` gets code `i + 1`, and a missing or unknown value gets code 0. The embedding
width is `min(16, (cardinality + 1) // 2)`.

To change the transform of every column of one kind in one group, add an entry to `group_overrides` in
`TRANSFORM_POLICY`. The cached transforms record the policy, so they are refitted on the next run.

## Caches to rebuild

Each stage caches its output under `data/processed/{NAME}/`, and only the last cache checks itself against the
earlier ones:

| Cache | Holds | Rebuilt when |
|---|---|---|
| `{Survey}/` (for example `CMAPSurvey/`) | the locations, journeys, and users | the directory is deleted |
| `Graph/` | the nodes and edges, with their features | the directory is deleted |
| `Tensors/` | the tensors, `schema.json`, and `fingerprint.json` | the directory is deleted |
| `Datasets/` | the split indices and the fitted transforms | the tensor fingerprint or the policy changes |

After a change to a stage, delete the cache of that stage and of every stage after it, up to `Tensors/`. The
`Datasets/` caches rebuild on their own, because the tensor fingerprint holds a hash of the schema. Each section below
names the caches to delete.

## Add a location feature

A location feature is a numeric column on the zones. The graph builder turns every column of the zones into a
feature, so a location feature needs no model code.

1. **Compute the column.** A feature for every dataset goes in `_build_graph` (`activitygraphs/graph/build.py`), next
   to the Overture POI counts and land uses. A feature from one dataset's own census goes in the `add_statistics` of
   that survey's module. The zones arrive in a projected UTM CRS, so areas and lengths come out in metres.
2. **Keep the feature comparable across cities.** Prefer a density or a share to a raw count, as
   `normalise_statistics` does for the population and the jobs. A raw count grows with the size of a zone, and the
   zones of the surveys differ in size. A feature that every survey can compute keeps the datasets ready for joint
   training.
3. **Fill every value.** The column must be numeric, with no missing value. `check_tensors` raises on a NaN or an
   infinite value, so fill a zone with no data with 0, as the land uses are filled.
4. **Exclude the helper columns.** Every column not listed in `COLS_EXCLUDED_FROM_FEATURES`
   (`activitygraphs/tensors/build.py`) becomes a feature. Add a helper column that must not reach the model, such as an
   identifier or an intermediate area, to that list.
5. **Check the inferred kind.** The kind of each location column is inferred by `infer_schema`. A column holding only
   0 and 1 becomes `binary`, and any other numeric column becomes `continuous`. Each inferred kind is logged as a
   warning on the first build. A heavy-tailed column, such as a raw count, scales better as a `count`. To get the
   log1p transform, pass `{"column_name": ColumnKind.COUNT}` as the overrides of the `infer_schema` call in
   `_build_schema`.
6. **Rebuild the caches.** Delete `Graph/` and `Tensors/`. The survey cache can stay, because the survey is unchanged.

`LocationFeatures` reads the whole of `locations.x`, and `_encoded_scorer` sizes each network from the encoder width.
The new column therefore reaches every variant with no change to the models. Compare the new inputs against the old
ones on `experiment=core`, the short run for paired comparisons between code changes.

## Add an edge feature

An edge feature is a column on the edges of the graph. The only edge feature today is `weight`, the distance in metres
between the centroids of the two zones, computed by `c2g.contiguity_graph`.

1. **Compute the column** on `edges_gdf` in `_build_graph`. The edges from `build_extra_edges` are concatenated after
   the contiguity edges, so compute the column after the concatenation, or make every extra-edges builder return it
   too. An extra edge without the column holds a NaN, and `check_tensors` raises.
2. **Declare the column** in `EDGE_SCHEMA` (`activitygraphs/tensors/build.py`), with its kind. The edge schema is
   declared rather than inferred, and `gdf_to_pyg` reads exactly the columns of `EDGE_SCHEMA`.
3. **Rebuild the caches.** Delete `Graph/` and `Tensors/`.

The edge width reaches the networks through `schema.edges.width`, which `_encoded_scorer` passes as `edge_dim`. The
GAT and the GPS layers read the edge features, and the node-wise MLP ignores them.

## Add a user attribute

A user attribute is a column of the users table, and the demographics are shared across the surveys. A new attribute
therefore touches the shared schema and every survey that has a person table.

1. **Declare the column** in `USER_SCHEMA` (`activitygraphs/base.py`), with its polars dtype.
2. **Declare its kind** in `DEMOGRAPHICS_SCHEMA` (`activitygraphs/surveys/demographics.py`). A categorical column
   takes its categories from a constant tuple in that module, such as `AGE_BANDS`, so that a category gets the same
   code in every dataset.
3. **Map the column in every survey with a person table.** Today these are `_build_cmap_users` in
   `activitygraphs/surveys/cmap.py` and `_build_tts_users` in `activitygraphs/surveys/tts.py`. Map each survey's own
   codes onto the shared categories. TPG has no person table, so its schema stays empty.
4. **Handle the missing answers.** A missing categorical answer stays null, which is encoded as code 0. A continuous
   column has no missing code, and `check_tensors` raises on a NaN, so a continuous attribute needs a value for every
   user. A count with an open top, such as the vehicle count, is usually better as bands, as `vehicle_band` does.
5. **Rebuild the caches.** The users table is built with the survey, so delete the whole `data/processed/{NAME}/`
   directory of every dataset with a person table.

The new column reaches the model through the `Demographics` source of `use_demographics`, and through the user tower of
the two-tower model. Test the mapping in each survey's test module. The test
`test_every_users_df_column_is_a_feature_or_an_identifier` in `test/test_tts.py` checks that every column of the users
table is either an identifier or a column of `DEMOGRAPHICS_SCHEMA`.

## Add an input source

An input source is a block of columns that a variant turns on with a flag of `InputFeatures`. The distance from home
(`use_dist`), the home-anchored PE (`use_home_pe`), the random-walk PE (`use_rwpe`), and the demographics
(`use_demographics`) are the existing sources. The running example is a source called `HomeLandUse`, with flag
`use_home_land_use`. It gives every node of a user the land-use shares of that user's home zone.

### 1. Compute the raw tensor

The raw tensor is built on the device at one of two places, depending on its shape:

- **A per-node encoding of the graph**, the same for every user, goes in `Locations.from_tensors`
  (`activitygraphs/datasets/locations.py`), as `random_walk_pe` and `hop_distance` do. Add a field to the `Locations`
  dataclass and document its shape under `Attributes`. `Locations.to` moves every tensor field to the device, so the
  new field needs no extra code there.
- **A (user, node) feature derived from the home node** goes in `UserBatch.join` (`activitygraphs/datasets/batch.py`).
  Index a `[num_nodes, num_nodes]` tensor of the locations by `home_node`, as `home_hops` does with
  `locations.hop_distance[home_node]`. Add a field to the `UserBatch` dataclass and document its shape.

A source that only reads existing tensors skips this step. The running example reads the land-use columns of
`locations.x` at each user's home node, so it needs no new tensor.

The `user_nodes` group holds `is_home` alone, on purpose, and `UserNodeFeatures` raises on any other column. A new
(user, node) feature is a new source rather than a new column of `user_nodes`.

### 2. Choose how the source is scaled

The feature transforms only scale the tensor groups of `DatasetSchema`. A new source has two options:

- **The source scales itself.** `RandomWalkPE` applies a batch norm to each column, and `HomeAnchoredPE` expands the
  hop distance into radial basis functions whose values lie between 0 and 1. This option needs no change to the
  schema, the cache, or the transforms.
- **The source reads a scaled tensor.** `HomeDistance` reads `batch.home_distance`, which is `distance_m` scaled by the
  transform of the `distances` group. A new group of this kind is a larger change: a field of `DatasetSchema`, a width
  in `check_tensors`, a fitted value in `fit_transforms`, and the application of the transform in
  `Locations.from_tensors`.

Prefer the first option unless the source needs a transform fitted on the training users.

### 3. Write the source

The source is a `FeatureSource` subclass in `activitygraphs/models/features.py`. It sets `out_dim` in its constructor,
from the schema or the config, and its `forward(locations, batch)` returns `[num_users, num_nodes, out_dim]`:

```python
class HomeLandUse(FeatureSource):
    """Extracts the land uses of the user's home zone, for every node of each user.

    Args:
        columns: Indices of the land-use columns in ``locations.x``.
    """

    def __init__(self, columns: list[int]):
        super().__init__()
        self.columns = columns
        self.out_dim = len(columns)

    def forward(self, locations: Locations, batch: UserBatch) -> Tensor:
        """Return the land uses of each user's home zone at every node, ``[num_users, num_nodes, out_dim]``."""
        home_features = locations.x[batch.home_node]
        home_land_use = home_features[:, self.columns]

        return home_land_use.unsqueeze(1).expand(-1, batch.num_nodes, -1)
```

Derive every width and column index from the schema, never from a constant. `TensorSchema.index_of(name)` returns the
index of one column, and `TensorSchema.names` lists every column in order. The number of location features differs
between datasets, so a hardcoded index breaks silently on another dataset.

### 4. Add the flag

Add a boolean flag to `InputFeatures`, with a default of `False`, and document it under `Attributes`:

```python
@dataclass(frozen=True)
class InputFeatures:
    ...
    use_home_land_use: bool = False
```

A default of `False` keeps the inputs and the names of the existing variants unchanged. It also keeps `HomeZoneMF`
valid, because `_build_home_zone_mf` rejects any flag that differs from its default.

### 5. Append the source to the encoder

`build_feature_encoder` appends one source per flag that is set. Append the new source after the last existing one, so
that the existing columns keep their positions, and add it to the column order listed in the docstring:

```python
    if features.use_home_land_use:
        land_use_columns = [idx for idx, name in enumerate(schema.locations.names) if name.startswith("land_use_")]
        sources.append(HomeLandUse(land_use_columns))
```

A source with its own settings, such as the `bins` and `max_hops` of the home-anchored PE, reads them from the config.
`build_scorer` passes only `train.home_pe` to the builders today. A new config block therefore needs a dataclass in
`activitygraphs/config.py`, a block in `activitygraphs/conf/config.yaml`, and a new argument of `build_scorer`, of
`ScorerBuilder`, and of every builder.

### 6. Name and log the flag

The reported name of a variant records every flag that differs from its default. Add a short suffix for the flag to
`FEATURE_SUFFIXES` (`activitygraphs/experiments/variants.py`), and add the flag to `variant_hyperparams`, so that W&B
logs it as a column. The suffix marks the flag when it differs from its default, here when it is on:

```python
FEATURE_SUFFIXES = {
    ...
    "use_home_land_use": "hlu",
}
```

The suffixes appear in the order of the fields of `InputFeatures`, so the suffix of a flag added last comes last.

### 7. Support the flag in the other scorers

Every network built by `_encoded_scorer` supports the new flag through `build_feature_encoder`. A scorer with its own
inputs supports the flag only once its builder handles it. `_build_two_tower` maps each flag to one of the two-tower
inputs, so it either maps the new flag too, or raises a `ValueError` when the flag is set. A flag that has no effect
on a scorer must never reach that scorer silently, because the variant name would report it.

### 8. Test the source

Three tests need the new flag:

- In `test/test_input_snapshot.py`, add the flag to `FLAGS`. The snapshot then covers every combination of the flags.
  Regenerate the snapshot with `uv run python test/test_input_snapshot.py`, and check that every existing combination
  is unchanged. Regenerate it only when the change to the inputs is intended.
- In `TestBuildScorer` of `test/test_experiments.py`, add a width test on the pattern of `test_home_pe_adds_its_bins`,
  which checks that the flag adds `out_dim` columns to the encoder.
- In `TestVariant` of the same module, check the new suffix in the variant name. `test_every_flag_has_a_suffix` fails
  until the flag has a suffix.

Write unit tests for the source itself on the stand-in graphs of `test/graphs.py` and the encoders of
`test/encoders.py`, which need no data.

### 9. Train with the source

The new source is trained once an experiment has a variant with the flag. The guide
[Adding an experiment](adding-an-experiment.md) shows how to write one, for example the entry:

```yaml
    - { model: mlp, features: { use_home_land_use: true } }
```

Read the new variant against the same variant without the flag, paired per user, rather than against the aggregate
means.
