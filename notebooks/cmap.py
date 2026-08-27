import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import torch
    import torch_geometric as pyg
    import polars as pl
    import geopandas as gpd

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root, data="cmap")
    return cfg, mo, project_root


@app.cell
def _(cfg, project_root):
    from activitygraphs.data.cmap import load_files

    inputs = load_files(cfg.data, project_root)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Decisions:
    - added hh_id columns to user_id
    - days table is not needed
    - rolled hh level attributes to user-level table (de-normalised)
    - num of observed days: always 1 => rolled into NetworkData for CMAP
    """)
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.dataprocessing import load_data

    cmap_data, network_nodes, network_edges = load_data(cfg.data, project_root)
    return


if __name__ == "__main__":
    app.run()
