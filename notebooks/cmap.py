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
    return network_edges, network_nodes


@app.cell
def _(network_edges, network_nodes):
    from activitygraphs.mapping import explore_locations_by_type, explore_walk_edges

    _m = network_nodes.explore()
    _m = network_edges.explore(m=_m)

    _m
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.ml.dataset import load_or_build_dataset

    tensors = load_or_build_dataset(cfg, project_root)
    return (tensors,)


@app.cell
def _(tensors):
    tensors.median_realised_size / len(tensors)
    return


@app.cell
def _(tensors):
    from activitygraphs.analysis import home_zone_summary

    home_zone_summary(tensors)
    return


@app.cell
def _(cfg):
    from activitygraphs.ml.datamodule import ActivityDataModule

    datamodule = ActivityDataModule(cfg, cfg.train.val_size, cfg.train.test_size, cfg.train.split_seed, cfg.train.batch_size)
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(datamodule):
    full_dataset = datamodule.train_dataset
    full_dataset.spatial_features[:, :, full_dataset.is_home_spatial_idx].argmax(dim=1).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
