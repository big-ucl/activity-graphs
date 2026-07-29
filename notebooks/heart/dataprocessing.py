import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)
    return (gva_data,)


@app.cell
def _(network_nodes):
    network_nodes
    return


@app.cell
def _(gva_data):
    from activitygraphs.dataprocessing import load_gva_network_graph

    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return network_edges, network_nodes


@app.cell(hide_code=True)
def _():
    run_graph_gen = mo.ui.run_button(kind="warn", label="Load PyG Graphs")
    run_graph_gen
    return (run_graph_gen,)


@app.cell
def _(gva_data, network_edges, network_nodes, run_graph_gen):
    from activitygraphs.dataprocessing import load_pyg_graphs

    mo.stop(not run_graph_gen.value)

    graphs = load_pyg_graphs(gva_data, network_nodes, network_edges, cfg.data, project_root)
    graphs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Visualising network and individual graphs
    """)
    return


@app.cell(hide_code=True)
def _(gva_data):
    from activitygraphs.dataprocessing import add_user_cols

    user_ids = gva_data.with_filter("subsector").user_ids
    select_user_id = mo.ui.dropdown(user_ids, value=user_ids[0], label="User ID:", searchable=True)
    return add_user_cols, select_user_id


@app.cell
def _(add_user_cols, gva_data, network_nodes, select_user_id):
    indiv_nodes = add_user_cols(select_user_id.value, network_nodes, gva_data.location_visits, gva_data.home_locations,
                                gva_data.work_locations, gva_data.edu_locations)
    indiv_nodes
    return (indiv_nodes,)


@app.cell(hide_code=True)
def _(indiv_nodes):
    select_node_col = mo.ui.dropdown(list(indiv_nodes.columns), searchable=True, label="Column:", value="purpose")
    return (select_node_col,)


@app.cell(hide_code=True)
def _():
    toggle_polygons = mo.ui.switch(value=False, label="Show subsectors")
    return (toggle_polygons,)


@app.cell(hide_code=True)
def _(
    indiv_nodes,
    network_edges,
    select_node_col,
    select_user_id,
    toggle_polygons,
):
    _nodes = indiv_nodes.set_geometry("original_geometry") if toggle_polygons.value else indiv_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(m=_m, column=select_node_col.value, marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Sampling and visualising predictions
    """)
    return


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
def _(network_nodes):
    network_nodes.sort_index()
    return


@app.cell
def _(
    network_edges,
    preds,
    select_node_col,
    select_user_id,
    toggle_polygons,
    user_nodes,
):
    _pred_nodes = user_nodes.copy().sort_index()
    _pred_nodes["predictions"] = preds

    nodes = _pred_nodes.set_geometry("original_geometry") if toggle_polygons.value else _pred_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(m=_m, column="predictions", marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return (nodes,)


@app.cell
def _():
    import contextily as cx

    return (cx,)


@app.cell
def _(CRS, cx, nodes):
    def plot_preds(nodes, col="predictions"):
        nodes = nodes.to_crs(CRS)

        ax = nodes.plot(column=col, figsize=(15, 15), legend=True, cmap="OrRd")
        nodes.boundary.plot(ax=ax, color="lightgrey")

        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    plot_preds(nodes)
    return (plot_preds,)


@app.cell
def _(plot_preds):
    plot_preds()
    return


if __name__ == "__main__":
    app.run()
