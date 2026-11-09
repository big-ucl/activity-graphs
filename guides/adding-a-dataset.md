# Adding a dataset

To add a new dataset to the `activity-graphs` package, you need two new code files:

- a new data config file
- a survey-specific python module, which turns the raw files into a subclass of `SurveyData`, a common interface for all
  later stages
  The graph, the tensors, the splits, and the feature transforms are then built from the `SurveyData` with no
  dataset-specific code, apart from the census statistics and any extra edges. The census statistics and extra edges on
  the graph are also handled in the survey-specific python module

_Note:_ All paths below are relative to the repository root.

In this guide, we follow a hypothetical running example: a new survey called XYZ. We create files:

- `activitygraphs/conf/data/xyz.yaml`, the config file, and
- `activitygraphs/surveys/xyz.py`, the survey-specific module, which defines classes:
    - `XYZSurvey`,
    - `XYZTables`, and
    - `XYZDataConfig`.
      The TTS module,  `activitygraphs/surveys/tts.py`, is the best template to copy from, because it has a person
      table, a
      household table, a census file, and island zones.

## 1. Place the raw files

Survey data is not distributed with the repository, so the raw files are not tracked in git. Each file goes under
`data/`, which is listed in `.gitignore`:

```
data/
  raw/XYZ/                              # the survey tables: trips, persons, households
  external/
    boundaries/<region or country>/     # the zone shapefiles, shared by the surveys of one country
    statistics/XYZ/                     # the census population and jobs files
    overture/XYZ/                       # land_use.geojson and place.geojson
```

The Overture files do not need to be fetched by hand. When `land_use.geojson` or `place.geojson` is missing,
`Overture.load` downloads both files with _city2graph_ for the area covered by the zones, and saves them to the
configured directory.

## 2. Declare the config dataclasses

The data config is typed by dataclasses in `activitygraphs/config.py`. Each dataset declares its own input files, so
add four dataclasses in the matching `# =====` sections of that module:

```python
@dataclass
class XYZBoundaryPaths:
    """Boundary files of the XYZ survey zones."""

    directory: str
    zones: str


@dataclass
class XYZStatsPaths(StatsPaths):
    """Census statistics file config for the XYZ survey."""

    population: str
    jobs: str


@dataclass
class XYZPaths(SurveyPaths):
    """Raw file inputs for the XYZ survey."""

    statistics: XYZStatsPaths
    boundaries: XYZBoundaryPaths

    raw_person: str
    raw_household: str


@dataclass
class XYZDataConfig(DataConfig):
    """Dataset config for the XYZ survey."""

    inputs: XYZPaths  # pyrefly: ignore[bad-override-mutable-attribute]
```

A field that varies between configs of the same survey goes on the `DataConfig` subclass. For example,
`TTSDataConfig.survey_version` selects the 2016 or the 2022 tables of TTS, so one survey module serves both
`tts2016.yaml` and `tts2022.yaml`.

Finally, add the yaml name `"xyz"` to `VALID_CONFIGS` at the bottom of `config.py`. The `read_config` helper and the
test fixtures accept only the names in `VALID_CONFIGS`.

## 3. Write the data config

The data config lives in `activitygraphs/conf/data/xyz.yaml`, and it must have exactly the fields of `XYZDataConfig`.
Start from a copy of `cmap.yaml` and edit it:

```yaml
# @package data
name: XYZ
survey: xyz

inputs:
  raw_journeys: trips.csv
  raw_person: persons.csv
  raw_household: households.csv

  statistics:
    directory: ${...paths.external}/statistics/${...name}
    population: population.csv
    jobs: jobs.csv

  overture:
    directory: ${...paths.external}/overture/${...name}
    land_use: land_use.geojson
    place: place.geojson

  boundaries:
    directory: ${...paths.external}/boundaries/<country>
    zones: zones.zip

paths:
  processed: data/processed/${..name}
  raw: data/raw/${..name}

  external: data/external
  gtfs: ${..paths.external}/gtfs

# Upper edges in km of the distance-from-home bands the within-band AUC is read on.
distance_bands: [ 2, 5, 10, 25 ]

# Share of the training batches run each epoch, drawn from a fresh shuffle of the training users every epoch.
train_batch_fraction: 1.0
```

Three keys need a decision for each dataset:

- `name` names the processed cache directory, the run directories (`reports/runs/XYZ-{N}-{timestamp}/`), and the W&B
  group.
- `survey` is the key of the survey loader in the registry (step 7). Two configs may share one survey key, as the two
  TTS configs do.
- `distance_bands` sets the bands of the within-band AUC. Pick edges that match the size of the region: TPG uses
  `[1, 2.5, 5, 10]` for Geneva, and CMAP uses `[2, 5, 10, 25]` for Chicago.

The `train_batch_fraction` key shortens each epoch on a large survey. TTS has about 200,000 users, so its configs
run a quarter of the training batches per epoch.

## 4. Write the survey module

The survey module `activitygraphs/surveys/xyz.py` builds three tables: the locations, the journeys, and the users. The
schemas of all three are declared in `activitygraphs/base.py`, and each builder ends with `check_schema`, which raises
on a missing column or a wrong dtype.

### 4.1. Read the raw tables

The raw files are read once into a frozen dataclass, so the builders below can be tested on synthetic tables. Write an
`XYZTables` dataclass that holds the parsed raw frames and the zone boundaries, and a public
`read_tables(cfg, project_root)` that reads them from the paths of the config. `read_tables` in `tts.py` shows how to
force the dtype of code columns with `schema_overrides`.

### 4.2. Build the locations

The locations are a GeoDataFrame with the columns of `LOCATIONS_SCHEMA`: `loc_id`, `loc_name`, `type`, `lon`, `lat`, and
`geometry`, in the CRS `EPSG:4326`. Each zone of the study area gets `type="zone"`, because the pipeline keeps only the
`zone` locations as graph nodes.

The locations also hold the `NA` sink, built by `build_special_locations()`. A trip end that falls outside the zones,
or that has no usable zone code, is coded to the `NA` location. Its `type` is `na`, so the graph and the tensors never
see it. `_zone_to_loc_id` in `tts.py` shows the mapping.

The `loc_id` values are strings, and the nodes are ordered by `loc_id` as strings. `"1001"` therefore comes before
`"625"`. Tests that check node indices must follow the same order.

### 4.3. Build the journeys

The journeys have one row per trip leg, with the columns of `USER_JOURNEY_SCHEMA`. A survey that records one leg per
trip, as TTS and CMAP do, sets `leg_id` to 1 for every row. A column that the survey does not record is filled with a
typed null, for example `dep_day=pl.lit(None, dtype=pl.Date)`.

The survey codes are mapped onto the shared `Mode` and `Purpose` enums of `activitygraphs/base.py`. Write the mapping as
a dict from each enum member to its survey codes, and invert it with `invert_mapping`. Map any other code to
`Mode.UNKNOWN` or `Purpose.UNKNOWN` with `replace_strict(..., default=...)`.

`Purpose.HOME` needs care. The base `SurveyData` finds each user's home from the trip ends with the home purpose, and it
drops a user with no home zone or with more than one. A survey with a household table overrides this rule in step 4.4.

### 4.4. Build the users

The users table depends on whether the survey has person and household tables. There are two cases:

- **No person table (TPG).** Do not build a users table. The base class derives each user's home from the home-purpose
  visits, uses `user_id` as `hh_id`, and declares no demographics.
- **Person and household tables (CMAP, TTS).** Build a users table with the columns of `USER_SCHEMA`: the identifiers
  `user_id`, `hh_id`, and `home_loc_id`, and the demographic columns.

The demographics are shared across the surveys. Each category is declared once in
`activitygraphs/surveys/demographics.py`, so a category gets the same code in every dataset. Map the survey's own codes
onto those categories, and leave a missing answer as null. A null is encoded as code 0. Two helpers in that module cover
the common cases:

- `quantile_bands` maps the income bands of the survey to within-survey income quartiles.
- `vehicle_band` maps a vehicle count to the bands `0`, `1`, `2`, and `3+`.

A survey that does not record one of the demographic columns fills the column with nulls of its `USER_SCHEMA` dtype.

The household id `hh_id` controls the split. `HouseholdSplit` keeps every member of a household in the same split, so
give co-residents the same `hh_id`, and give every other user a distinct one.

### 4.5. Write the survey class

The survey class subclasses `SurveyData` for the interface and `DataFrameStore` for the cache layout. It implements
`_copy`, `load`, and `save`. A survey with a users table also overrides `users_df`, `home_locations`, `user_ids`, and
`demographics_schema`:

```python
class XYZSurvey(SurveyData, DataFrameStore):
    """XYZ ``SurveyData`` subclass with caching via ``DataFrameStore``."""

    def __init__(
            self,
            locations_gdf: gpd.GeoDataFrame,
            user_journeys_df: pl.DataFrame,
            users_df: pl.DataFrame,
            filters: list[str] | None = None,
    ):
        super().__init__(user_journeys_df, locations_gdf, filters)
        self._users_df = users_df

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        """Home location of each user, one row per user."""
        return self.users_df.select("user_id", loc_id="home_loc_id")

    @cached_property
    def users_df(self) -> pl.DataFrame:
        """Users, filtered by active home location filters, sorted by ``user_id``."""
        return self._filter_loc_types(self._users_df, "home_loc_id").sort("user_id")

    @cached_property
    def user_ids(self) -> pl.Series:
        """Sorted ids of the users in ``users_df``."""
        return self.users_df["user_id"].sort()

    @property
    def demographics_schema(self) -> TensorSchema:
        """Schema of the demographic columns in ``users_df``, in tensor column order."""
        return DEMOGRAPHICS_SCHEMA

    def _copy(self, filters: list[str] | None = None) -> Self:
        return type(self)(self._locations_gdf, self._user_journeys_df, self.users_df, filters)
```

Copy `load` and `save` from `TTSSurvey`. `load` reads the three parquet files from the processed cache when the cache
directory exists. Otherwise, it calls `read_tables` and `_build_xyz_survey`, and saves the result.

## 5. Add the census statistics and the extra edges

The graph builder adds census and Overture features to the zones, and it needs two functions from the survey module
for the census part:

- `read_statistics(cfg, project_root)` reads the census files and returns them in any type. Wrap the census tables in
  a small frozen dataclass, such as `TTSStatistics`.
- `add_statistics(zones_gdf, statistics)` returns the zones with a `population` column, a `jobs` column where the
  census has jobs, and an `area` column. The zones arrive in a projected UTM CRS, with a `loc_id` column. Use
  `overlap_counts` to split each census unit's count by its share of area in each zone, and `normalise_statistics` to
  turn the counts into densities.

Every column added to the zones becomes a location feature, except the columns listed in
`COLS_EXCLUDED_FROM_FEATURES` in `activitygraphs/tensors/build.py`. The kind of each feature is inferred by
`infer_schema`, which logs a warning per inferred column. Read those warnings on the first build to check that each
column got the kind you expected.

The contiguity graph leaves a zone with no touching neighbour unconnected. An island or a detached zone is the usual
case. Write `build_extra_edges(nodes_gdf, edges_gdf)` to join such zones to the graph, or pass `None` when every zone
has a neighbour. `_build_tts_island_edges` joins each isolated zone to its three nearest zones with
`c2g.knn_graph`, and returns an empty frame when there is nothing to join.

## 6. Bundle the loader

The survey module ends with a `LOADER`, the `SurveyLoader` that bundles the steps above:

```python
LOADER = SurveyLoader(
    load=XYZSurvey.load,
    read_statistics=_read_xyz_statistics,
    add_statistics=_add_xyz_population_job_statistics,
    build_extra_edges=_build_xyz_island_edges,
)
```

Keep the module layout of the other surveys: the public classes and functions first, then the private helpers, in
sections split by `# =====` banners. The `LOADER` reads the helpers at import time, so it goes below the helpers it
names.

## 7. Register the survey

The registry maps the `survey` key of a data config to its loader. Import the new module in
`activitygraphs/surveys/registry.py` and add an entry to `REGISTRY`:

```python
from activitygraphs.surveys import cmap, thats, tpg, tts, xyz

REGISTRY: dict[str, SurveyLoader[Any, Any]] = {
    ...
"xyz": xyz.LOADER,
}
```

Also add `xyz` to the list of survey modules in the docstring of `activitygraphs/surveys/__init__.py`.

## 8. Write the tests

The unit tests run without any data on disk, so they build the survey from synthetic tables. Two existing tests need a
new entry:

- In `test/test_config.py`, add `"xyz": XYZDataConfig` to `DATA_CONFIGS`. The test then checks that `xyz.yaml` has
  exactly the fields of `XYZDataConfig`.
- `test/test_surveys.py` reads `VALID_CONFIGS`, so it checks the new survey key against the registry with no change.

Write `test/test_xyz.py` on the pattern of `test/test_tts.py`. Small factories (`make_trips`, `make_persons`,
`make_households`, `make_tables`) build `XYZTables` from a few rows, and each test checks one mapping or one filter.
Cover at least the following:

- each builder returns a frame that passes its schema;
- every survey code maps to the intended `Mode`, `Purpose`, or demographic category, and an unknown code maps to the
  unknown or missing value;
- a trip end outside the zones becomes the `NA` sink, and a user with no home zone is dropped; and
- co-residents share a group id, and the home nodes and visit labels come out at the expected node indices.

A test that needs the real data takes the `load_or_skip` fixture from `test/conftest.py` and the
`@pytest.mark.integration` marker. Such a test is skipped when the data is absent. Run both sets:

```bash
uv run pytest test/
uv run pytest test/ -m integration
```

## 9. Build the caches and run the baselines

The first run builds every cache of the dataset under `data/processed/XYZ/`:

```
data/processed/XYZ/
  XYZSurvey/      # the locations, journeys, and users
  Graph/          # nodes.parquet and edges.parquet
  Tensors/        # the location and user tensors, schema.json, and fingerprint.json
  Datasets/       # the split indices and the fitted feature transforms
```

Start with a single-batch run, which builds the caches and runs every stage once without writing a run directory:

```bash
uv run activity-graphs data=xyz trainer.fast_dev_run=true train.wandb=false
```

The survey, graph, and tensor caches are read whenever their files exist, with no check against the code. After a
change to the survey module, delete `data/processed/XYZ/` so that every stage is rebuilt. The `Datasets/` caches are
checked against the tensor fingerprint, and they rebuild on their own when the tensors change.

Once the caches are built, fit the baselines alone and read the report:

```bash
uv run activity-graphs data=xyz experiment=baselines
uv run report-results data=xyz
```

The baselines are a sanity check on the new dataset. Every popularity and distance baseline should score well
above `Uniform`. A baseline at the level of `Uniform` points to a wrong home zone, wrong visit labels, or a wrong node
order. The report also prints the home-zone occupancy: the number of users per home zone, and the share of users
whose home zone holds at least 10, 25, or 60 users.

## 10. Update the documentation

The README lists the datasets and the `data=` options in its Usage section. Add the new dataset to both, so that
`uv run activity-graphs data=xyz` is documented next to `tpg` and `cmap`.
