import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    from activitygraphs.ml.dataset import load_gva_dataset

    test_size = 0.2
    seed = 42

    train_dataset, test_dataset = load_gva_dataset(cfg, test_size, seed)
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    from activitygraphs.experiments import build_gat, build_mlp

    gat = build_gat(train_dataset, 8, 128, 0.2)
    gat.load_state_dict(torch.load(project_root / "models" / "GATSkip-8-res.pth", weights_only=True))

    mlp = build_mlp(train_dataset, 3, 128, 0.2)
    mlp.load_state_dict(torch.load(project_root / "models" / "MLP.pth", weights_only=True))

    gat, mlp
    return (gat,)


@app.cell
def _():
    reports_path = project_root / cfg.paths.reports / "data"
    figures_path = project_root / cfg.paths.figures
    return figures_path, reports_path


@app.cell
def _(reports_path):
    results = pl.read_parquet(reports_path / "geneva-results.parquet")
    results
    return (results,)


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(alt, figures_path, results):
    _results = results.filter(pl.col("epoch") > 5).unpivot(
        on=["bce_weight"],
        index=["name", "epoch"],
        value_name="BCE",
        variable_name="Dataset",
    )

    fig = (
        alt
        .Chart(_results)
        .mark_line(point=True)
        .encode(
            x="epoch",
            y="BCE",
            color="name",
            facet="Dataset",
            tooltip=["name", "epoch", "BCE"],
        )
    )

    fig.save(figures_path / "results.png")

    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Visualising prediction on the map
    """)
    return


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData
    from activitygraphs.dataprocessing import load_gva_network_graph

    gva_data = GenevaData.load(cfg.data, project_root)
    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return gva_data, network_edges, network_nodes


@app.cell
def _(test_dataset):
    index = 1
    data = test_dataset[index]
    user_id = data.user_id
    data
    return (data,)


@app.cell
def _(data, gat):
    _batch = next(iter(pyg.loader.DataLoader([data])))
    preds = torch.sigmoid(gat(_batch.x, _batch.edge_index, _batch.edge_attr, _batch.batch)).detach().cpu().numpy()
    preds.T
    return (preds,)


@app.cell
def _(gva_data):
    from activitygraphs.dataprocessing import add_user_cols

    user_ids = gva_data.with_filter("subsector").user_ids
    select_user_id = mo.ui.dropdown(user_ids, value=user_ids[0], label="User ID:", searchable=True)
    toggle_polygons = mo.ui.switch(value=False, label="Show subsectors")
    return add_user_cols, select_user_id, toggle_polygons


@app.cell
def _(add_user_cols, gva_data, network_nodes, select_user_id):
    indiv_nodes = add_user_cols(select_user_id.value, network_nodes, gva_data.location_visits, gva_data.home_locations,
                                gva_data.work_locations, gva_data.edu_locations)
    indiv_nodes
    return (indiv_nodes,)


@app.cell
def _(indiv_nodes):
    select_node_col = mo.ui.dropdown(list(indiv_nodes.columns), searchable=True, label="Column:", value="purpose")
    return (select_node_col,)


@app.cell
def _(
    indiv_nodes,
    network_edges,
    preds,
    select_node_col,
    select_user_id,
    toggle_polygons,
):
    _pred_nodes = indiv_nodes.copy().sort_index()
    _pred_nodes["predictions"] = preds

    nodes = _pred_nodes.set_geometry("original_geometry") if toggle_polygons.value else _pred_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(m=_m, column="predictions", marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


if __name__ == "__main__":
    app.run()
