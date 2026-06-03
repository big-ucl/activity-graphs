import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root, data="toronto")


@app.cell
def _():
    CRS = "EPSG:4326"
    return


@app.cell
def _():
    import geopandas as gpd
    import polars as pl

    return (pl,)


@app.cell
def _():
    from activitygraphs.data.toronto import TorontoData
    from activitygraphs.dataprocessing import load_toronto_network_graph, convert_to_torch

    from activitygraphs.ml.dataset import ActivityDataset

    return (
        ActivityDataset,
        TorontoData,
        convert_to_torch,
        load_toronto_network_graph,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data processing
    """)
    return


@app.cell
def _(
    ActivityDataset,
    TorontoData,
    convert_to_torch,
    load_toronto_network_graph,
):
    data = TorontoData.load(cfg.data, project_root).with_filter("subsector")

    network_nodes, network_edges = load_toronto_network_graph(
        data, cfg.data, project_root
    )

    network_graph, spatial_features, spatial_labels, demographics, distances = convert_to_torch(data, network_nodes, network_edges)

    dataset_path = project_root / cfg.data.paths.pyg_datasets

    dataset = ActivityDataset(dataset_path, network_graph, spatial_features, spatial_labels, demographics, distances)
    return data, dataset, network_edges, network_nodes


@app.cell
def _(data):
    data.inputs.raw_person_df
    return


@app.cell
def _(data, pl):
    cols = [f"THATS D{n}DS? = 1" for n in range(1, 8)]
    obs = data.inputs.raw_person_df.select("person_id", pl.sum_horizontal(cols).alias("num_obs_days"), "THATS Number of Trips", span=pl.col("THATS Experiment start day") - pl.col("THATS Experiment end day"))
    ttrips = data.user_journeys_df.group_by("user_id").agg(n_trip_days=pl.col("dep_day").n_unique().cast(pl.Int32))
    locs = data.location_visits.group_by("user_id").agg(num_locs=pl.col("loc_id").n_unique())


    res = ttrips.join(obs, left_on="user_id", right_on="person_id").with_columns(day_diff=pl.col("n_trip_days") - pl.col("num_obs_days")).join(locs, on="user_id")
    return obs, res


@app.cell
def _(alt, res):
    alt.Chart(res).mark_bar().encode(x=alt.X("n_trip_days").bin(maxbins=50), y="count()")
    return


@app.cell
def _(alt, obs):
    alt.Chart(obs).mark_point().encode(x="num_obs_days", y="THATS Number of Trips")
    return


@app.cell
def _(alt, res):
    alt.Chart(res).mark_bar().encode(x="num_obs_days:N", y="count()")
    return


@app.cell
def _(pl, res):
    res.filter(pl.col("n_trip_days") <= 7, day_diff=0)
    return


@app.cell
def _(alt, res):
    alt.Chart(res).mark_point().encode(x="n_trip_days", y="THATS Number of Trips").properties(width=1000)
    return


@app.cell
def _(alt, res):
    alt.Chart(res).mark_point().encode(x="n_trip_days", y="num_locs")
    return


@app.cell
def _(alt, res):
    alt.Chart(res).mark_point().encode(x="num_obs_days", y="num_locs")
    return


@app.cell
def _(data):
    data.users_df
    return


@app.cell
def _(data, pl):
    data.user_journeys_df.group_by("user_id", maintain_order=True).agg(n_trip_days=pl.col("dep_day").unique())
    return


@app.cell
def _(dataset):
    from torch_geometric.loader import DataLoader

    dataset.num_classes
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The PyG Dataset is composed of three objects:
    1. network graph (PyG) - simple city2graph call on network_nodes & network_edges
    2. user spatial features (Tensor, dim N_users x N_nodes x 1) - is_home indicator, to be concatenated with NG.x in dataset
    3. user features (Tensor, dim N_users x N_features) - socio-demographics about users, can be concatenated with each row of NG.x
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualisation
    """)
    return


@app.cell
def _(network_nodes):
    select_node_col = mo.ui.dropdown(
        list(network_nodes.columns), searchable=True, label="Column:"
    )
    return (select_node_col,)


@app.cell
def _(network_edges, network_nodes, select_node_col):
    _nodes = network_nodes  # .set_geometry("original_geometry") #if toggle_polygons.value else indiv_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(
        m=_m, column=select_node_col.value, marker_kwds={"radius": 5}
    )

    mo.vstack([
        select_node_col,  # mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _():
    from activitygraphs.base import Mode
    from activitygraphs.data.toronto import MANUAL_MODE_MAP, MODE_MAP

    return MANUAL_MODE_MAP, MODE_MAP, Mode


@app.cell
def _():
    import itertools

    return (itertools,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trip modes

    First: investigate labeling behaviour
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Proportion of unlabeled trips
    """)
    return


@app.cell(hide_code=True)
def _(data):
    data.inputs.raw_journeys_df.select(
        "predicted_modes", "manual_mode"
    ).null_count() / len(data.inputs.raw_journeys_df)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    How often do people label their trips?
    """)
    return


@app.cell
def _(alt, data, pl):
    _mode_label_freq = data.inputs.raw_journeys_df.group_by("app_id").agg(
        pct_labeled=pl.col("manual_mode").is_not_null().sum() / pl.len()
    )

    _chart = _mode_label_freq.plot.bar(
        x=alt.X("pct_labeled", bin=alt.Bin(maxbins=20)), y="count()"
    ).properties(
        title="Distribution of proportion of manually-labeled trip modes per user"
    )
    _chart
    return


@app.cell(hide_code=True)
def _(alt, itertools, pl):
    def plot_enum_confusion_matrix(
        df: pl.DataFrame, pred_col: str, actual_col: str, enum, scale_type="linear"
    ) -> alt.Chart:
        width, height = 400, 400

        base = pl.DataFrame(
            list(itertools.product(enum, enum)),
            orient="row",
            schema=[pred_col, actual_col],
        )

        modes = df.group_by(pred_col, actual_col).agg(count=pl.len())
        modes = base.join(modes, on=[pred_col, actual_col], how="left").fill_null(
            0
        )
        modes = modes.sort(pred_col, actual_col)

        row_totals = modes.group_by(actual_col, maintain_order=True).agg(
            pl.col("count").sum()
        )
        col_totals = modes.group_by(pred_col, maintain_order=True).agg(
            pl.col("count").sum()
        )

        labels = (
            alt
            .Chart(modes)
            .mark_text()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                text=alt.Text("count:Q"),
            )
        )

        boxes = (
            alt
            .Chart(modes)
            .mark_rect()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                color=alt.Color(
                    "count:Q", scale=alt.Scale(type=scale_type, domainMin=1)
                ),
                tooltip=[pred_col, actual_col, "count"],
            )
        )

        row_boxes = (
            alt
            .Chart(row_totals)
            .mark_rect()
            .encode(
                y=alt.Y(f"{actual_col}:N", axis=None),
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(type=scale_type, domainMin=1),
                ),
            )
        )

        row_labels = (
            alt
            .Chart(row_totals)
            .mark_text()
            .encode(
                y=alt.Y(
                    f"{actual_col}:N",
                ),
                text=alt.Text("count:Q"),
            )
        )

        col_boxes = (
            alt
            .Chart(col_totals)
            .mark_rect()
            .encode(
                x=alt.X(f"{pred_col}:N", axis=None),
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(type=scale_type, domainMin=1),
                ),
            )
        )

        col_labels = (
            alt
            .Chart(col_totals)
            .mark_text()
            .encode(
                x=alt.X(
                    f"{pred_col}:N",
                ),
                text=alt.Text("count:Q"),
            )
        )

        main = (boxes + labels).properties(width=width, height=height)
        rows = (row_boxes + row_labels).properties(width=30, height=height)
        cols = (col_boxes + col_labels).properties(width=width, height=30)

        bottom = (main | rows).resolve_scale(y="shared")

        return (
            (cols & bottom)
            .resolve_scale(x="shared", color="shared")
            .properties(
                title="Confusion matrix of GPS predicted vs manually-entered modes",
            )
        )

    return (plot_enum_confusion_matrix,)


@app.cell(hide_code=True)
def _():
    drop_down_log_scale = mo.ui.dropdown(
        ["log", "linear"], label="Color scale", value="linear"
    )
    return (drop_down_log_scale,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparison between predicted mode and manually-entered mode. In genereal, it's somewhat confused.
    """)
    return


@app.cell(hide_code=True)
def _(
    MANUAL_MODE_MAP,
    MODE_MAP,
    Mode,
    data,
    drop_down_log_scale,
    pl,
    plot_enum_confusion_matrix,
):
    _mode_enum = [m for m in Mode if m not in [Mode.COACH]]
    _df = data.inputs.raw_journeys_df.select(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),
        pl.col("manual_mode").replace_strict(
            MANUAL_MODE_MAP, default=Mode.UNKNOWN
        ),
    )

    _chart = plot_enum_confusion_matrix(
        _df,
        "predicted_modes",
        "manual_mode",
        Mode,
        scale_type=drop_down_log_scale.value,
    )

    mo.vstack([drop_down_log_scale, _chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Purposes

    - Link purposes from activities to trips, then compare with accuracy from manually labeled trip purposes
    """)
    return


@app.cell
def _(data):
    from activitygraphs.data.toronto import build_toronto_activities

    activs = build_toronto_activities(data.inputs, data.user_journeys_df)
    print(activs.head())
    return


@app.cell
def _(data, pl):
    def compare_trip_to_activity(
        activities: pl.DataFrame,
        trips: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Join trips to the collapsed activity that 'receives' each trip
        (i.e. the activity whose end_trip_id matches the trip_id, or
        failing that, the activity happening at the arrival hour with
        the closest purpose match).

        Returns the trips frame with extra columns:
          - act_purpose        : purpose from the activity log
          - purpose_match      : bool, whether arr_purpose == act_purpose
          - match_method       : how the match was made
        """

        # Combine trip legs into single trips
        trip_summaries = (
            trips
            .sort("user_id", "journey_id", "leg_id")
            .group_by("user_id", "journey_id", maintain_order=True)
            .agg(
                dep_day=pl.col("dep_day").first(),
                dep_time=pl.col("dep_time").first(),
                dep_purpose=pl.col("dep_purpose").first(),
                arr_purpose=pl.col("arr_purpose").last(),
                last_leg_dep=pl.col("dep_time").last(),
                last_leg_duration=pl.col("duration").last(),
                arr_day=pl.col("dep_day").last(),
            )
            .with_columns(
                arr_time=(
                    pl.col("arr_day").dt.combine(pl.col("last_leg_dep"))
                    + pl.col("last_leg_duration")
                ),
            )
            .with_columns(
                arr_day=pl.col("arr_time").dt.date(),
                arr_time=pl.col("arr_time").dt.time(),
            )
            .drop("last_leg_dep", "last_leg_duration")
        )

        # Join trips to activities where the arrival falls within the span.
        # Since act_hour has 1hr resolution, an arrival at e.g. 16:30 should
        # match an activity spanning 16:00–18:00.  We truncate arrival time
        # to the hour for comparison.
        matched = (
            trip_summaries
            .with_columns(arr_hour=pl.col("arr_time").dt.hour())
            .join(
                activities,
                left_on=["user_id", "arr_day"],
                right_on=["person_id", "act_date"],
                how="left",
                suffix="_act",
            )
            .filter(
                (pl.col("arr_hour") >= pl.col("start_time").dt.hour())
                & (pl.col("arr_hour") <= pl.col("end_time").dt.hour())
            )
            .with_columns(
                purpose_match=(pl.col("arr_purpose") == pl.col("act_purpose")),
                start_hour=pl.col("start_time").dt.hour(),
            )
            # Prefer: 1) purpose match, 2) latest start time (most recent activity)
            .sort(
                "user_id",
                "journey_id",
                "purpose_match",
                "start_hour",
                descending=[False, False, True, True],
            )
            .group_by("user_id", "journey_id")
            .first()
        )

        # Re-attach trips that had no matching activity at all
        unmatched = trip_summaries.join(
            matched.select("user_id", "journey_id"),
            on=["user_id", "journey_id"],
            how="anti",
        ).with_columns(
            act_purpose=pl.lit(None, dtype=pl.Categorical),
            purpose_match=pl.lit(None, dtype=pl.Boolean),
            # Add columns that the matched frame has so concat works
            start_time=pl.lit(None, dtype=pl.Time),
            end_time=pl.lit(None, dtype=pl.Time),
            n_hours=pl.lit(None, dtype=pl.UInt32),
            hh_id=pl.lit(None, dtype=pl.Utf8),
            arr_hour=pl.lit(None, dtype=pl.Int8),
        )

        return pl.concat([matched, unmatched], how="diagonal").sort(
            "user_id", "dep_day", "dep_time"
        )


    compared = compare_trip_to_activity(data.activities_df, data.user_journeys_df)
    compared
    return (compared,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - What proportion of unlabled trip purposes actually have corresponding data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Rule finding ML algorithm (perhaps)
    """)
    return


@app.cell
def _(data):
    trips = data.user_journeys_df
    activities = data.activities_df
    return activities, trips


@app.cell
def _(activities, trips):
    trip_users = trips.select("user_id").unique()
    act_users = activities.select("person_id").unique()
    print("Trip users:", trip_users.height)
    print("Activity persons:", act_users.height)
    print(
        "Overlap:",
        trip_users.join(
            act_users, left_on="user_id", right_on="person_id", how="inner"
        ).height,
    )
    return


@app.cell
def _(activities, trips):
    # Check 2: Do the dates overlap?
    trip_dates = trips.select("dep_day").unique()
    act_dates = activities.select("act_date").unique()
    print("Trip dates:", trip_dates.height)
    print("Activity dates:", act_dates.height)
    print(
        "Date overlap:",
        trip_dates.join(
            act_dates, left_on="dep_day", right_on="act_date", how="inner"
        ).height,
    )
    return


@app.cell
def _(activities, trips):
    # Per-user date coverage
    trip_user_dates = (
        trips.select("user_id", "dep_day").unique().rename({"dep_day": "date"})
    )
    act_user_dates = (
        activities
        .select("person_id", "act_date")
        .unique()
        .rename({"person_id": "user_id", "act_date": "date"})
    )

    per_user = (
        trip_user_dates
        .join(act_user_dates, on=["user_id", "date"], how="inner")
        .group_by("user_id")
        .len()
    )
    print(f"Users with at least 1 overlapping date: {per_user.height}")
    print(f"Mean overlapping dates per user: {per_user['len'].mean():.1f}")
    print(f"Users with 0 overlapping dates: {588 - per_user.height}")
    return (act_user_dates,)


@app.cell
def _(pl, trips):
    trip_summaries = (
        trips
        .sort("user_id", "journey_id", "leg_id")
        .group_by("user_id", "journey_id", maintain_order=True)
        .agg(
            dep_day=pl.col("dep_day").first(),
            dep_time=pl.col("dep_time").first(),
            dep_purpose=pl.col("dep_purpose").first(),
            arr_purpose=pl.col("arr_purpose").last(),
            last_leg_dep=pl.col("dep_time").last(),
            last_leg_duration=pl.col("duration").last(),
            arr_day=pl.col("dep_day").last(),
        )
        .with_columns(
            arr_time=(
                pl.col("arr_day").dt.combine(pl.col("last_leg_dep"))
                + pl.col("last_leg_duration")
            ),
        )
        .with_columns(
            arr_day=pl.col("arr_time").dt.date(),
            arr_time=pl.col("arr_time").dt.time(),
        )
        .drop("last_leg_dep", "last_leg_duration")
    )
    return (trip_summaries,)


@app.cell
def _(act_user_dates, compared, pl, trip_summaries):
    # Trips that are actually matchable (user + date exists in activities)
    matchable_trips = trip_summaries.join(
        act_user_dates,
        left_on=["user_id", "arr_day"],
        right_on=["user_id", "date"],
        how="inner",
    )
    print(f"Matchable trips: {matchable_trips.height}")

    matchable_compared = compared.join(
        matchable_trips.select("user_id", "journey_id"),
        on=["user_id", "journey_id"],
        how="inner",
    )
    total = matchable_compared.height
    matched = matchable_compared.filter(pl.col("act_purpose").is_not_null()).height
    purpose_matched = matchable_compared.filter(
        pl.col("purpose_match") == True
    ).height
    print(f"Matchable trips: {total}")
    print(f"Matched to activity: {matched} ({matched / total:.1%})")
    print(
        f"Purpose agrees: {purpose_matched} ({purpose_matched / max(matched, 1):.1%} of matched)"
    )

    # How many matchable trips have null arr_hour?
    null_arr = matchable_trips.filter(pl.col("arr_time").is_null()).height
    print(
        f"Matchable but null arr_hour: {null_arr} ({null_arr / matchable_trips.height:.1%})"
    )
    return (matchable_compared,)


@app.cell
def _(matchable_compared, pl):
    # What are the most common mismatches?
    mismatches = (
        matchable_compared
        .filter(pl.col("purpose_match") == False)
        .group_by("arr_purpose", "act_purpose")
        .len()
        .sort("len", descending=True)
    )
    print("Top 15 purpose mismatches:")
    print(mismatches.head(15))

    # How many trip purposes are already unknown?
    print(
        matchable_compared.select(
            arr_null=pl.col("arr_purpose").is_null().sum(),
            arr_unknown=(pl.col("arr_purpose").cast(pl.Utf8) == "unknown").sum(),
            act_null=pl.col("act_purpose").is_null().sum(),
            act_unknown=(pl.col("act_purpose").cast(pl.Utf8) == "unknown").sum(),
        )
    )
    return


if __name__ == "__main__":
    app.run()
