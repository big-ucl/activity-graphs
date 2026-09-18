from pathlib import Path

import marimo

from activitygraphs.analysis import load_score_vectors

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Activity graphs - results and analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset selection
    """)
    return


@app.cell
def _(mo):
    dataset_dropdown = mo.ui.dropdown(["cmap", "geneva"], value="cmap", label="Dataset:")
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(dataset_dropdown):
    dataset_str = dataset_dropdown.value
    full_run = 7
    demographics_run = 4
    overfit_run = 5
    latest = None

    run = latest
    extra_runs = []
    return dataset_str, extra_runs, overfit_run, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports, config, and loading
    """)
    return


@app.cell
def _(dataset_str, mo):
    import altair as alt
    import polars as pl

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root, data=dataset_str)
    return alt, cfg, pl, project_root


@app.cell
def _():
    # Categorical slots in fixed order, chosen for colour-blind separation; never reordered or cycled.
    SERIES_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    LEARNED_HUE, BASELINE_HUE = SERIES_HUES[0], "#898781"
    AHEAD_HUE, BEHIND_HUE = SERIES_HUES[0], SERIES_HUES[7]
    GRID_HUE = "#898781"
    return (
        AHEAD_HUE,
        BASELINE_HUE,
        BEHIND_HUE,
        GRID_HUE,
        LEARNED_HUE,
        SERIES_HUES,
    )


@app.cell
def _(pl):
    def json_safe(frame: pl.DataFrame) -> pl.DataFrame:
        """Replace all NaNs with None in DataFrame. Avoids crashes due to JSON conversion in Altair."""
        return frame.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))

    return (json_safe,)


@app.cell
def _():
    from activitygraphs.analysis import POPULARITY_MODEL, R_PRECISION_COLUMN, load_report_run, load_run
    from activitygraphs.analysis import _latest_run as latest_run

    return POPULARITY_MODEL, R_PRECISION_COLUMN, latest_run, load_report_run, load_run


@app.cell
def _(cfg, extra_runs, latest_run, load_report_run, project_root, run):
    reports_data_path = project_root / cfg.paths.reports_data
    ranking_budget = cfg.train.largest_size_recall_at_k
    report_run = latest_run(reports_data_path, cfg.data.name) if run is None else run

    aggregate_results, per_user_results = load_report_run(
        reports_data_path, cfg.data.name, report_run, extra_runs, ranking_budget
    )
    return (
        aggregate_results,
        per_user_results,
        ranking_budget,
        report_run,
        reports_data_path,
    )


@app.cell
def _(cfg):
    analysis = cfg.analysis
    return (analysis,)


@app.cell
def _(aggregate_results, pl):
    model_kinds = (
        aggregate_results
        .filter(pl.col("stage") == "test")
        .group_by("name")
        .agg(kind=pl.when(pl.col("seed").is_null().all()).then(pl.lit("baseline")).otherwise(pl.lit("learned")))
    )

    model_kinds
    return (model_kinds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aggregate results
    """)
    return


@app.cell
def _(analysis, ranking_budget):
    headline = f"{analysis.per_user_metric}@{ranking_budget}"
    return (headline,)


@app.cell(hide_code=True)
def _(headline, mo):
    mo.md(f"""
    Metric: _{headline}_, the per-user mean recall@k over k = 1..K, averaged over users and then over seeds. It is
    derived from the stored per-positive ranks, so runs recorded before it was logged read on it too.
    """)
    return


@app.cell
def _(analysis, per_user_results, pl):
    from activitygraphs.analysis import PER_USER_STAGE, aggregate_metrics, per_user_metric_summary

    aggregate_table = per_user_metric_summary(per_user_results, analysis.per_user_metric)
    headline_per_seed = (
        per_user_results
        .filter(pl.col("stage") == PER_USER_STAGE)
        .group_by("name", "seed")
        .agg(value=pl.col(analysis.per_user_metric).mean())
    )
    aggregate_table
    return aggregate_metrics, aggregate_table, headline_per_seed


@app.cell(hide_code=True)
def _(analysis, headline, mo):
    mo.md(f"""
    Comparison chart between all models (learned and baselines) on _{headline}_. 

    - Each seed (model run) is a hollow circle.
    - The mean score is the filled circle, along with a +- 1 sd interval
    - Baselines are deterministic (single point, no interval)
    - Vertical dashed rule shows the reference model: _{analysis.reference_model}_
    """)
    return


@app.cell(hide_code=True)
def _(
    BASELINE_HUE,
    GRID_HUE,
    LEARNED_HUE,
    aggregate_table,
    alt,
    analysis,
    headline,
    headline_per_seed,
    json_safe,
    model_kinds,
    pl,
):
    _scores_per_seed = json_safe(headline_per_seed.join(model_kinds, on="name"))
    _score_spread = json_safe(
        aggregate_table.join(model_kinds, on="name").with_columns(
            lo=pl.col("mean") - pl.col("sd").fill_null(0.0),
            hi=pl.col("mean") + pl.col("sd").fill_null(0.0),
        )
    )

    _y = alt.Y("name:N", sort=aggregate_table["name"].to_list(), title=None)
    _colour = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=["learned", "baseline"], range=[LEARNED_HUE, BASELINE_HUE]),
        legend=alt.Legend(title=None, orient="top"),
    )

    _score_points_chart = (
        alt
        .Chart(_scores_per_seed)
        .mark_point(filled=False, size=45, strokeWidth=1.5, opacity=0.7)
        .encode(y=_y, x="value:Q", color=_colour, tooltip=["name", "seed", "value"])
    )
    _score_interval_chart = (
        alt
        .Chart(_score_spread)
        .mark_rule(strokeWidth=2, opacity=0.45)
        .encode(y=_y, x=alt.X("lo:Q", title=headline), x2="hi:Q", color=_colour)
    )
    _score_mean_chart = (
        alt
        .Chart(_score_spread)
        .mark_point(filled=True, size=90, opacity=1.0)
        .encode(
            y=_y,
            x=alt.X("mean:Q", title=headline, scale=alt.Scale(zero=True)),
            color=_colour,
            tooltip=["name", "n_users", "n_seeds", "mean", "sd"],
        )
    )
    _reference_model_rule = (
        alt
        .Chart(_score_spread.filter(pl.col("name") == analysis.reference_model))
        .mark_rule(color=GRID_HUE, strokeDash=[4, 4])
        .encode(x="mean:Q")
    )

    alt.layer(_reference_model_rule, _score_interval_chart, _score_points_chart, _score_mean_chart).properties(
        width=540,
        height=alt.Step(22),
        title=f"{headline} by model, mean +- 1 sd over seeds",
    )
    return


@app.cell(hide_code=True)
def _(R_PRECISION_COLUMN, mo):
    mo.md(f"""
    Diagnostic: _{R_PRECISION_COLUMN}_, how often the top-|RG_i| picks are right, as logged at test time.
    """)
    return


@app.cell
def _(R_PRECISION_COLUMN, aggregate_metrics, aggregate_results):
    aggregate_metrics(aggregate_results, R_PRECISION_COLUMN)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-user paired comparison
    """)
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    Paired per-user _{analysis.per_user_metric}_, difference compaired to reference _{analysis.reference_model}_ (avg over seeds)
    """)
    return


@app.cell
def _(analysis, per_user_results):
    from activitygraphs.analysis import paired_comparison

    paired_metrics = paired_comparison(per_user_results, analysis.reference_model, analysis.per_user_metric)
    paired_metrics
    return paired_comparison, paired_metrics


@app.cell(hide_code=True)
def _(AHEAD_HUE, BEHIND_HUE, GRID_HUE, alt, json_safe, pl):
    def plot_paired_comparison(paired_metrics: pl.DataFrame, per_user_metric: str, reference_model: str) -> alt.Chart:
        # Setup
        _model_diffs = json_safe(
            paired_metrics.with_columns(
                direction=pl
                .when(pl.col("mean_diff") >= 0)
                .then(pl.lit("ahead of reference"))
                .otherwise(pl.lit("behind reference"))
            )
        )
        _y = alt.Y("name:N", sort=_model_diffs["name"].to_list(), title=None)
        _colour = alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["ahead of reference", "behind reference"], range=[AHEAD_HUE, BEHIND_HUE]),
            legend=alt.Legend(title=None, orient="top"),
        )
        _x_title = f"paired difference in {per_user_metric} vs {reference_model}"

        # Sub charts
        _zero_rule_chart = alt.Chart(_model_diffs).mark_rule(color=GRID_HUE, strokeDash=[4, 4]).encode(x=alt.datum(0))

        _interval_chart = (
            alt
            .Chart(_model_diffs)
            .mark_rule(strokeWidth=2, opacity=0.45)
            .encode(y=_y, x=alt.X("ci_lo:Q", title=_x_title), x2="ci_hi:Q", color=_colour)
        )
        _point_chart = (
            alt
            .Chart(_model_diffs)
            .mark_point(filled=True, size=90, opacity=1.0)
            .encode(
                y=_y,
                x=alt.X("mean_diff:Q", title=_x_title),
                color=_colour,
                tooltip=["name", "n_users", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "per_seed_sd"],
            )
        )

        return alt.layer(_zero_rule_chart, _interval_chart, _point_chart).properties(
            width=540,
            height=alt.Step(22),
            title="Paired per-user difference, 95% bootstrap interval over users",
        )

    return (plot_paired_comparison,)


@app.cell
def _(analysis, paired_metrics, plot_paired_comparison):
    plot_paired_comparison(paired_metrics, analysis.per_user_metric, analysis.reference_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparison with MLP
    """)
    return


@app.cell
def _(analysis, paired_comparison, per_user_results):
    paired_metrics_mlp = paired_comparison(per_user_results, reference_model="MLP-dist", metric=analysis.per_user_metric)
    paired_metrics_mlp
    return (paired_metrics_mlp,)


@app.cell
def _(analysis, paired_metrics_mlp, plot_paired_comparison):
    plot_paired_comparison(paired_metrics_mlp, analysis.per_user_metric, "MLP-dist")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Realised set size
    Test users by home-included `|RG_i|`. A user whose only visit is their own home has nothing left to rank once home
    is excluded, so they are dropped from the per-user metrics.
    """)
    return


@app.cell
def _(aggregate_results, per_user_results):
    from activitygraphs.analysis import _num_scored_dropped_users, realised_size_summary

    _, _n_dropped = _num_scored_dropped_users(aggregate_results)

    realised_sizes = realised_size_summary(per_user_results, _n_dropped)
    realised_sizes
    return (realised_sizes,)


@app.cell(hide_code=True)
def _(BASELINE_HUE, LEARNED_HUE, alt, json_safe, pl, realised_sizes):
    _statuses = ["scored", "dropped (home only)"]
    _sizes = json_safe(
        realised_sizes.with_columns(
            status=pl.when("scored").then(pl.lit(_statuses[0])).otherwise(pl.lit(_statuses[1])),
            label=(100 * pl.col("share_of_users")).round(1).cast(pl.Utf8) + pl.lit("%"),
        )
    )

    _size_bar = (
        alt
        .Chart(_sizes)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("n_pos_home_incl:O", title="|RG_i| including home", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share_of_users:Q", title="share of test users", axis=alt.Axis(format="%")),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=_statuses, range=[LEARNED_HUE, BASELINE_HUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["n_pos_home_incl", "status", "n_users", "share_of_users"],
        )
    )
    _size_labels = _size_bar.mark_text(dy=-6).encode(text="label:N", color=alt.value("#52514e"))

    alt.layer(_size_bar, _size_labels).properties(width=540, height=200, title="Test users by realised set size")
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    Per-user _{analysis.per_user_metric}_ restricted to the users with home-included `|RG_i| >= {analysis.min_realised_size}`,
    and the paired comparison against _{analysis.reference_model}_ on the same users.
    """)
    return


@app.cell
def _(analysis, paired_comparison, per_user_metric_summary, per_user_results):
    from activitygraphs.analysis import _restrict_to_realised_size

    _restricted = _restrict_to_realised_size(per_user_results, analysis.min_realised_size)
    paired_metrics_restricted = paired_comparison(_restricted, analysis.reference_model, analysis.per_user_metric)

    per_user_metric_summary(_restricted, analysis.per_user_metric)
    return (paired_metrics_restricted,)


@app.cell
def _(paired_metrics_restricted):
    paired_metrics_restricted
    return


@app.cell
def _(analysis, paired_metrics_restricted, plot_paired_comparison):
    plot_paired_comparison(paired_metrics_restricted, analysis.per_user_metric, analysis.reference_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Home coverage
    Share of test users with a home node that has no training user. Same paired comparison split by `home_seen_in_train`.
    """)
    return


@app.cell
def _(per_user_results):
    from activitygraphs.analysis import home_coverage_summary

    home_coverage = home_coverage_summary(per_user_results)
    home_coverage
    return (home_coverage,)


@app.cell(hide_code=True)
def _(analysis, per_user_results, pl):
    from activitygraphs.analysis import paired_comparison_by_home_coverage

    paired_comparison_home_cov = paired_comparison_by_home_coverage(
        per_user_results, analysis.reference_model, analysis.per_user_metric
    )

    paired_comparison_seen_home = paired_comparison_home_cov.filter("home_seen_in_train").drop("home_seen_in_train")
    paired_comparison_unseen_home = paired_comparison_home_cov.filter(~pl.col("home_seen_in_train")).drop(
        "home_seen_in_train"
    )
    return (
        paired_comparison_home_cov,
        paired_comparison_seen_home,
        paired_comparison_unseen_home,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Home seen in training:
    """)
    return


@app.cell
def _(paired_comparison_seen_home):
    paired_comparison_seen_home
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Home not seen in training
    """)
    return


@app.cell
def _(paired_comparison_unseen_home):
    paired_comparison_unseen_home
    return


@app.cell(hide_code=True)
def _(
    GRID_HUE,
    SERIES_HUES,
    alt,
    analysis,
    home_coverage,
    json_safe,
    paired_comparison_home_cov,
    paired_metrics,
    pl,
):
    _strata = ["home in train", "home unseen"]
    _stratum = pl.when("home_seen_in_train").then(pl.lit(_strata[0])).otherwise(pl.lit(_strata[1])).alias("stratum")
    _share = json_safe(
        home_coverage.with_columns(_stratum, label=(100 * pl.col("share")).round(1).cast(pl.Utf8) + pl.lit("%"))
    )
    _by_stratum = json_safe(paired_comparison_home_cov.with_columns(_stratum))

    _colour = alt.Color(
        "stratum:N",
        scale=alt.Scale(domain=_strata, range=[SERIES_HUES[0], SERIES_HUES[1]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    _y = alt.Y("name:N", sort=paired_metrics["name"].to_list(), title=None)
    _offset = alt.YOffset("stratum:N", sort=_strata)
    _x_title = f"paired difference in {analysis.per_user_metric} vs {analysis.reference_model}"

    _share_bar = (
        alt
        .Chart(_share)
        .mark_bar(height=26)
        .encode(
            x=alt.X("share:Q", title="share of test users", axis=alt.Axis(format="%")),
            color=_colour,
            order=alt.Order("stratum:N"),
            tooltip=["stratum", "n_users", "share"],
        )
    )
    _share_labels = _share_bar.mark_text(fontWeight="bold").encode(
        text="label:N", color=alt.value("white"), x=alt.X("share:Q", stack="zero", bandPosition=0.5)
    )

    _zero_chart = alt.Chart(_by_stratum).mark_rule(color=GRID_HUE, strokeDash=[4, 4]).encode(x=alt.datum(0))
    _interval_chart = (
        alt
        .Chart(_by_stratum)
        .mark_rule(strokeWidth=2, opacity=0.45)
        .encode(y=_y, yOffset=_offset, x=alt.X("ci_lo:Q", title=_x_title), x2="ci_hi:Q", color=_colour)
    )
    _point_chart = (
        alt
        .Chart(_by_stratum)
        .mark_point(filled=True, size=70, opacity=1.0)
        .encode(
            y=_y,
            yOffset=_offset,
            x="mean_diff:Q",
            color=_colour,
            tooltip=["name", "stratum", "n_users", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p"],
        )
    )

    alt.vconcat(
        alt.layer(_share_bar, _share_labels).properties(width=540, height=46, title="Test users by home coverage"),
        alt.layer(_zero_chart, _interval_chart, _point_chart).properties(
            width=540,
            height=30 * _by_stratum["name"].n_unique(),
            title="Paired difference within each home-coverage stratum",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hop bands analysis
    """)
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    ### Decomposition by hop band of each positive
    Every positive is ranked against the full candidate set and credited to the band it sits in, so a model's band
    contributions sum to its per-user _{analysis.per_user_metric}_.

    - `share_of_metric`: fraction of the model's _{analysis.per_user_metric}_ delivered by the band
    - `share_of_pos`: fraction of the positives that sit in the band
    """)
    return


@app.cell
def _(analysis, per_user_results):
    from activitygraphs.analysis import band_decomposition

    band_decomposition_table = band_decomposition(per_user_results, analysis.per_user_metric)
    band_decomposition_table
    return (band_decomposition_table,)


@app.cell(hide_code=True)
def _(SERIES_HUES, alt, analysis, band_decomposition_table, json_safe, pl):
    _band_order = band_decomposition_table["band"].unique(maintain_order=True).to_list()
    _models = (
        band_decomposition_table.group_by("name").agg(pl.col("mean").sum()).sort("mean", descending=True)["name"].to_list()
    )
    _contributions = json_safe(
        band_decomposition_table.with_columns(
            band_rank=pl.col("band").replace_strict({band: i for i, band in enumerate(_band_order)}, return_dtype=pl.Int64)
        )
    )

    (
        alt
        .Chart(_contributions)
        .mark_bar()
        .encode(
            y=alt.Y("name:N", sort=_models, title=None),
            x=alt.X("mean:Q", stack="zero", title=f"{analysis.per_user_metric}, mean over seeds"),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=_band_order, range=SERIES_HUES[: len(_band_order)]),
                legend=alt.Legend(title="hops from home", orient="top"),
            ),
            order=alt.Order("band_rank:Q"),
            tooltip=["name", "band", "mean", "sd", "share_of_metric", "share_of_pos"],
        )
        .properties(
            width=540,
            height=alt.Step(22),
            title=f"{analysis.per_user_metric} split by the hop band of each positive (bars sum to the headline)",
        )
    )
    return


@app.cell(hide_code=True)
def _(BASELINE_HUE, alt, band_decomposition_table, json_safe, pl):
    _band_order = band_decomposition_table["band"].unique(maintain_order=True).to_list()
    _positives = json_safe(
        band_decomposition_table
        .group_by("band")
        .agg(share_of_pos=pl.col("share_of_pos").mean())
        .with_columns(label=(100 * pl.col("share_of_pos")).round(1).cast(pl.Utf8) + pl.lit("%"))
    )
    _positives_bar = (
        alt
        .Chart(_positives)
        .mark_bar(color=BASELINE_HUE, opacity=0.55, cornerRadiusEnd=3)
        .encode(
            x=alt.X("band:N", sort=_band_order, title="hops from home", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share_of_pos:Q", title="share of positives", axis=alt.Axis(format="%")),
            tooltip=["band", "share_of_pos"],
        )
    )
    _positives_labels = _positives_bar.mark_text(dy=-6, color="#52514e").encode(text="label:N")

    alt.layer(_positives_bar, _positives_labels).properties(width=540, height=110, title="Where the positives are")
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    Paired per-user difference in each band's contribution against _{analysis.reference_model}_. A model's band
    differences sum to its overall paired difference, so this locates the distance rings in which it gains or loses.
    """)
    return


@app.cell
def _(analysis, per_user_results):
    from activitygraphs.analysis import paired_comparison_by_band

    paired_metrics_by_band = paired_comparison_by_band(
        per_user_results, analysis.reference_model, analysis.per_user_metric
    )
    paired_metrics_by_band
    return (paired_metrics_by_band,)


@app.cell(hide_code=True)
def _(
    AHEAD_HUE,
    BEHIND_HUE,
    GRID_HUE,
    alt,
    analysis,
    json_safe,
    paired_metrics,
    paired_metrics_by_band,
    pl,
):
    _band_order = paired_metrics_by_band["band"].unique(maintain_order=True).to_list()
    _by_band = json_safe(
        paired_metrics_by_band.with_columns(
            direction=pl
            .when(pl.col("mean_diff") >= 0)
            .then(pl.lit("ahead of reference"))
            .otherwise(pl.lit("behind reference"))
        )
    )
    _y = alt.Y("name:N", sort=paired_metrics["name"].to_list(), title=None)
    _colour = alt.Color(
        "direction:N",
        scale=alt.Scale(domain=["ahead of reference", "behind reference"], range=[AHEAD_HUE, BEHIND_HUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    _x_title = "paired difference"

    _zero_rule_chart = alt.Chart().mark_rule(color=GRID_HUE, strokeDash=[4, 4]).encode(x=alt.datum(0))
    _interval_chart = (
        alt
        .Chart()
        .mark_rule(strokeWidth=2, opacity=0.45)
        .encode(y=_y, x=alt.X("ci_lo:Q", title=_x_title), x2="ci_hi:Q", color=_colour)
    )
    _point_chart = (
        alt
        .Chart()
        .mark_point(filled=True, size=70, opacity=1.0)
        .encode(
            y=_y,
            x=alt.X("mean_diff:Q", title=_x_title),
            color=_colour,
            tooltip=["band", "name", "n_users", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "per_seed_sd"],
        )
    )

    (
        alt
        .layer(_zero_rule_chart, _interval_chart, _point_chart, data=_by_band)
        .properties(width=150, height=alt.Step(18))
        .facet(column=alt.Column("band:N", sort=_band_order, title="hops from home"))
        .properties(
            title=f"Paired difference in each band's contribution vs {analysis.reference_model}, "
            "95% bootstrap interval over users"
        )
    )
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    ## Recall@k curve
    Per-user recall at each cutoff k in `analysis.recall_curve_ks` = {list(analysis.recall_curve_ks)}, read from the
    number of candidates scored above and tied with each positive. A tied positive counts as a fractional hit: the share
    of its tied positions that fall inside the top-k. A model that scores every candidate alike therefore gets
    k / n_candidates rather than a perfect score.
    """)
    return


@app.cell(hide_code=True)
def _(POPULARITY_MODEL, mo):
    mo.md(f"""
    `lift` is each model's mean recall@k over that of _{POPULARITY_MODEL}_ at the same k, which is what compares across
    datasets: a fixed k covers a different share of each graph.
    """)
    return


@app.cell
def _(POPULARITY_MODEL, analysis, per_user_results):
    from activitygraphs.analysis import lift_over_reference

    recall_curve_table = lift_over_reference(per_user_results, POPULARITY_MODEL, analysis.recall_curve_ks)
    recall_curve_table
    return (recall_curve_table,)


@app.cell(hide_code=True)
def _(SERIES_HUES, aggregate_table, analysis, mo, recall_curve_table):
    _models = recall_curve_table["name"].unique(maintain_order=True).to_list()

    _ranked = [name for name in aggregate_table["name"] if name in _models]
    _defaults = [name for name in [*_ranked[:2], analysis.reference_model] if name in _models]

    recall_model_select = mo.ui.multiselect(
        options=_models,
        value=list(dict.fromkeys(_defaults)),
        label="models",
        max_selections=len(SERIES_HUES),
    )

    recall_model_select
    return (recall_model_select,)


@app.cell(hide_code=True)
def _(
    SERIES_HUES,
    alt,
    analysis,
    json_safe,
    pl,
    recall_curve_table,
    recall_model_select,
):
    _shown = [
        model for model in recall_curve_table["name"].unique(maintain_order=True) if model in recall_model_select.value
    ]
    _selected = json_safe(
        recall_curve_table.filter(pl.col("name").is_in(_shown)).with_columns(
            lo=pl.col("mean") - pl.col("sd").fill_null(0.0), hi=pl.col("mean") + pl.col("sd").fill_null(0.0)
        )
    )

    _ks = list(analysis.recall_curve_ks)
    _x = alt.X(
        "k:Q",
        title="k (choice-set size)",
        scale=alt.Scale(type="log", base=2, domain=[min(_ks), max(_ks)], nice=False, padding=12),
        axis=alt.Axis(values=_ks, format="d"),
    )
    _colour = alt.Color(
        "name:N",
        scale=alt.Scale(domain=_shown, range=SERIES_HUES[: len(_shown)]),
        legend=alt.Legend(title=None, orient="top", columns=2),
    )

    _interval = alt.Chart(_selected).mark_rule(strokeWidth=2, opacity=0.35).encode(x=_x, y="lo:Q", y2="hi:Q", color=_colour)
    _line = (
        alt
        .Chart(_selected)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=_x,
            y=alt.Y("mean:Q", title="recall@k", scale=alt.Scale(domain=[0, 1])),
            color=_colour,
            tooltip=["name", "k", "n_seeds", "mean", "sd", "lift"],
        )
    )

    alt.layer(_interval, _line).properties(width=540, height=300, title="Per-user recall@k, mean +- 1 sd over seeds")
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    Paired per-user recall@k against _{analysis.reference_model}_ at each cutoff, for the models selected above.
    """)
    return


@app.cell
def _(analysis, per_user_results):
    from activitygraphs.analysis import paired_comparison_by_k

    paired_metrics_by_k = paired_comparison_by_k(per_user_results, analysis.reference_model, analysis.recall_curve_ks)
    paired_metrics_by_k
    return (paired_metrics_by_k,)


@app.cell(hide_code=True)
def _(
    GRID_HUE,
    SERIES_HUES,
    alt,
    analysis,
    json_safe,
    paired_metrics_by_k,
    pl,
    recall_curve_table,
    recall_model_select,
):
    # Colour domain matches the recall@k chart above, so a model keeps its hue across both.
    _shown = [
        model for model in recall_curve_table["name"].unique(maintain_order=True) if model in recall_model_select.value
    ]
    _selected = json_safe(paired_metrics_by_k.filter(pl.col("name").is_in(_shown)))

    _ks = list(analysis.recall_curve_ks)
    _x = alt.X(
        "k:Q",
        title="k (choice-set size)",
        scale=alt.Scale(type="log", base=2, domain=[min(_ks), max(_ks)], nice=False, padding=12),
        axis=alt.Axis(values=_ks, format="d"),
    )
    _colour = alt.Color(
        "name:N",
        scale=alt.Scale(domain=_shown, range=SERIES_HUES[: len(_shown)]),
        legend=alt.Legend(
            title=None,
            orient="top",
            columns=2,
            values=[model for model in _shown if model != analysis.reference_model],
        ),
    )

    _zero_rule_chart = alt.Chart(_selected).mark_rule(color=GRID_HUE, strokeDash=[4, 4]).encode(y=alt.datum(0))
    _interval_chart = alt.Chart(_selected).mark_area(opacity=0.18).encode(x=_x, y="ci_lo:Q", y2="ci_hi:Q", color=_colour)
    _line_chart = (
        alt
        .Chart(_selected)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=_x,
            y=alt.Y("mean_diff:Q", title=f"paired difference in recall@k vs {analysis.reference_model}"),
            color=_colour,
            tooltip=["name", "k", "n_users", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "per_seed_sd"],
        )
    )

    alt.layer(_zero_rule_chart, _interval_chart, _line_chart).properties(
        width=540,
        height=260,
        title="Paired per-user recall@k difference, 95% bootstrap interval over users",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model health checks
    Checks that the models are behaving semi-correctly. Needs a run made with `train.save_score_vectors=true`.

    - `user_invariance` near 1.0: the model ranks every user identically, i.e. it has collapsed to one
      global ordering.
    - `popularity_corr` near 1.0: the model is stongly correlated with the popularity marginal, despite a better aggregate score.
    - `home_is_top1`: share of users whose own home node is ranked first. Also see  `user_invariance`: if the model does not differentiate well across users it will also be 1.0 there.
    """)
    return


@app.cell
def _(cfg, extra_runs, report_run, reports_data_path):
    path = Path(reports_data_path)
    scores = load_score_vectors(path, cfg.data.name, report_run)
    score_vectors = scores
    score_vectors
    return (score_vectors,)


@app.cell
def _(POPULARITY_MODEL, per_user_results, score_vectors):
    from activitygraphs.analysis import check_model_health

    model_health = check_model_health(score_vectors, per_user_results, POPULARITY_MODEL)
    model_health
    return (model_health,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overfit health
    Best `train_avg_recall@K` reached on the training batches. On a run made with
    `train.overfit_batches > 0` this asks whether a model can fit a batch it is allowed to memorise;
    on a normal run it is the train side of the train/val ranking gap. Needs
    `train.log_train_ranking=true`.
    """)
    return


@app.cell
def _(cfg, load_run, overfit_run, ranking_budget, reports_data_path):
    from activitygraphs.analysis import check_overfit_health

    _aggregate_results, _ = load_run(reports_data_path, cfg.data.name, run=overfit_run)

    overfit_health = check_overfit_health(_aggregate_results, ranking_budget)
    overfit_health
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Home zones analysis
    Based on original `ActivityDataset`, no model outputs needed.
    """)
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.ml.dataset import load_or_build_dataset

    dataset = load_or_build_dataset(cfg, project_root)
    return (dataset,)


@app.cell
def _(dataset):
    from activitygraphs.analysis import home_zone_summary

    home_zone_summary(dataset)
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    ## Summary
    One row per model: the headline metric, its paired difference against _{analysis.reference_model}_,
    and the health checks.
    """)
    return


@app.cell
def _(aggregate_table, model_health, paired_metrics):
    summary = (
        aggregate_table
        .select("name", "mean", "sd")
        .join(
            paired_metrics.select("name", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p"),
            on="name",
            how="left",
        )
        .join(
            model_health.select("name", "user_invariance", "popularity_corr", "home_is_top1"),
            on="name",
            how="left",
        )
        .sort("mean", descending=True)
    )
    summary
    return (summary,)


@app.cell
def _(
    BASELINE_HUE,
    LEARNED_HUE,
    alt,
    headline,
    json_safe,
    model_kinds,
    pl,
    summary,
):
    _measures = [headline, "paired difference", "user_invariance", "popularity_corr", "home_is_top1"]
    _long = json_safe(
        summary
        .select(
            "name",
            pl.col("mean").alias(headline),
            pl.col("mean_diff").alias("paired difference"),
            "user_invariance",
            "popularity_corr",
            "home_is_top1",
        )
        .unpivot(index="name", variable_name="measure", value_name="value")
        .drop_nulls("value")
        .drop_nans("value")
        .join(model_kinds, on="name")
    )

    (
        alt
        .Chart(_long)
        .mark_point(filled=True, size=70, opacity=1.0)
        .encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("name:N", sort=summary["name"].to_list(), title=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["learned", "baseline"], range=[LEARNED_HUE, BASELINE_HUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["name", "measure", "value"],
        )
        .properties(width=130, height=alt.Step(20))
        .facet(column=alt.Column("measure:N", sort=_measures, title=None))
        .resolve_scale(x="independent")
    )
    return


if __name__ == "__main__":
    app.run()
