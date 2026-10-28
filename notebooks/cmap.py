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

    from activitygraphs.config import read_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent)
    cfg = read_config(project_root, data="cmap")
    return cfg, mo, project_root


@app.cell
def _(cfg, project_root):
    from activitygraphs.surveys.cmap import read_tables

    inputs = read_tables(cfg.data, project_root)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Decisions:
    - added hh_id columns to user_id
    - days table is not needed
    - rolled hh level attributes to user-level table (de-normalised)
    - num of observed days: always 1 => rolled into SurveyData for CMAP
    """)
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.graph import load_graph
    from activitygraphs.surveys.registry import load_survey

    cmap_survey = load_survey(cfg.data, project_root).with_filter("zone")
    nodes_gdf, edges_gdf = load_graph(cmap_survey, cfg.data, project_root)
    return edges_gdf, nodes_gdf


@app.cell
def _(edges_gdf, nodes_gdf):
    from activitygraphs.mapping import explore_locations_by_type, explore_walk_edges

    _m = nodes_gdf.explore()
    _m = edges_gdf.explore(m=_m)

    _m
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.datasets.dataset import load_dataset

    tensors = load_dataset(cfg, project_root)
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
    from activitygraphs.datasets import HouseholdSplit, VisitDataModule

    datamodule = VisitDataModule(
        cfg,
        HouseholdSplit(cfg.train.val_size, cfg.train.test_size),
        cfg.train.split_seed,
        cfg.train.loader.batch_size,
        cfg.train.loader.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(datamodule):
    full_dataset = datamodule.train_dataset
    is_home_idx = full_dataset.schema.user_nodes.index_of("is_home")
    full_dataset.user_node_features[:, :, is_home_idx].argmax(dim=1).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
