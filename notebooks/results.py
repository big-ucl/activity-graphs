import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import torch
    import torch_geometric as pyg
    import polars as pl

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root, data="geneva")
    return cfg, mo, project_root


@app.cell
def _():
    from activitygraphs.analysis import load_run

    return (load_run,)


@app.cell
def _(cfg, load_run, project_root):
    reports_data_path = project_root / cfg.paths.reports_data

    aggregate_results, per_user_results = load_run(
        reports_data_path, cfg.data.name
    )
    return aggregate_results, per_user_results


@app.cell
def _(cfg):
    analysis = cfg.analysis
    return (analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aggregate results
    """)
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"""
    Metric: _{analysis.main_metric}_, averaged over seeds
    """)
    return


@app.cell
def _(aggregate_results, analysis):
    from activitygraphs.analysis import aggregate_metrics

    aggregate_metrics = aggregate_metrics(aggregate_results, analysis.main_metric)
    aggregate_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-user paired comparison
    """)
    return


@app.cell(hide_code=True)
def _(analysis, mo):
    mo.md(f"Paired per-user _{analysis.per_user_metric}_, difference compaired to reference _{analysis.reference_model}_ (avg over seeds)")
    return


@app.cell
def _(analysis, per_user_results):
    from activitygraphs.analysis import paired_comparison

    paired_metrics = paired_comparison(
        per_user_results, analysis.reference_model, analysis.per_user_metric
    )
    paired_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hop bands analysis
    """)
    return


@app.cell
def _(aggregate_results):
    from activitygraphs.analysis import compute_hop_band_table

    hop_bands = compute_hop_band_table(aggregate_results)
    hop_bands
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


@app.cell
def _(per_user_results):
    set(per_user_results["seed"].drop_nulls())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
