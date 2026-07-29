import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import polars as pl
    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    models_path = project_root / cfg.paths.models

    batch_size = 64
    test_size = 0.2
    seed = 42
    return batch_size, models_path, seed, test_size


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Loading the models
    """)
    return


@app.cell
def _():
    from activitygraphs.ml.dataset import load_gva_dataset
    from activitygraphs.experiments import build_gat, build_mlp

    return build_gat, build_mlp, load_gva_dataset


@app.cell
def _(batch_size, load_dataset, seed, test_size):
    train_set, test_set = load_dataset(cfg, test_size, seed)
    train_loader = pyg.loader.DataLoader(train_set, batch_size=batch_size)
    test_loader = pyg.loader.DataLoader(test_set, batch_size=batch_size)
    return test_set, train_loader, train_set


@app.cell
def _(build_gat, build_mlp, models_path, train_set):
    hidden_channels = 128
    dropout = 0.2
    gat_layers = 8
    gps_layers = 4
    mlp_layers = 3

    def load_model(file, builder, layers):
        path = models_path / file

        model = builder(train_set, layers, hidden_channels, dropout)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.cuda().eval()

    gat = load_model("GATSkip-8-res.pth", build_gat, gat_layers)
    mlp = load_model("MLP.pth", build_mlp, mlp_layers)
    gat, mlp
    return gat, mlp


@app.cell
def _(train_loader, train_set):
    from activitygraphs.ml.baselines import ConditionalNodeBaseline, NodeBaseline

    node_baseline = NodeBaseline(train_set[0].num_nodes).fit(train_loader)
    cond_baseline = ConditionalNodeBaseline(train_set[0].num_nodes).fit(train_loader)

    node_baseline, cond_baseline
    return cond_baseline, node_baseline


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Loading network data
    """)
    return


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData
    from activitygraphs.dataprocessing import load_gva_network_graph

    gva_data = GenevaData.load(cfg.data, project_root)
    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return gva_data, network_edges, network_nodes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Making predictions
    """)
    return


@app.cell
def _():
    "user_id = 25895"
    return


@app.cell
def _(test_set):
    index = 1
    data = test_set[index]
    user_id = data.user_id
    return data, user_id


@app.cell
def _():
    from activitygraphs.ml.sampling import poisson_sampling, pps_sampling

    return poisson_sampling, pps_sampling


@app.cell
def _(
    cond_baseline,
    data,
    gat,
    mlp,
    node_baseline,
    poisson_sampling,
    pps_sampling,
):
    batch = next(iter(pyg.loader.DataLoader([data]))).cuda()

    def predict_and_sample(model, batch, n=15):
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits)

        generator = None  # torch.Generator(device="cuda").manual_seed(seed)

        poisson = poisson_sampling(logits, generator)
        pps = pps_sampling(n, logits, batch.batch, generator)

        return logits, probs, poisson, pps

    logits, probs, poisson, pps = predict_and_sample(gat, batch)
    mlp_logits, mlp_probs, mlp_poisson, mlp_pps = predict_and_sample(mlp, batch)
    node_logits, node_probs, node_poisson, node_pps = predict_and_sample(node_baseline, batch)
    cond_logits, cond_probs, cond_poisson, cond_pps = predict_and_sample(cond_baseline, batch)
    return (
        cond_poisson,
        cond_pps,
        cond_probs,
        mlp_poisson,
        mlp_pps,
        mlp_probs,
        node_poisson,
        node_pps,
        node_probs,
        poisson,
        pps,
        probs,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Visualising predictions and samples
    """)
    return


@app.cell
def _():
    from activitygraphs.dataprocessing import add_user_cols

    return (add_user_cols,)


@app.cell
def _(
    add_user_cols,
    cond_poisson,
    cond_pps,
    cond_probs,
    gva_data,
    mlp_poisson,
    mlp_pps,
    mlp_probs,
    network_nodes,
    node_poisson,
    node_pps,
    node_probs,
    poisson,
    pps,
    probs,
    user_id,
):
    def add_preds_and_sample(nodes, preds, poisson, pps, prefix=""):
        nodes = nodes.copy().sort_index()

        nodes[prefix + "preds"] = preds.detach().cpu().numpy()
        nodes[prefix + "poisson"] = poisson.detach().cpu().numpy()
        nodes[prefix + "pps"] = pps.detach().cpu().numpy()

        return nodes

    user_nodes = add_user_cols(
        user_id,
        network_nodes,
        gva_data.location_visits,
        gva_data.home_locations,
        gva_data.work_locations,
        gva_data.edu_locations,
    )
    user_nodes = add_preds_and_sample(user_nodes, probs, poisson, pps)
    user_nodes = add_preds_and_sample(user_nodes, mlp_probs, mlp_poisson, mlp_pps, prefix="mlp_")
    user_nodes = add_preds_and_sample(user_nodes, node_probs, node_poisson, node_pps, prefix="node_")
    user_nodes = add_preds_and_sample(user_nodes, cond_probs, cond_poisson, cond_pps, prefix="cond_")

    user_nodes
    return (user_nodes,)


@app.cell
def _(user_nodes):
    select_node_col = mo.ui.dropdown(list(user_nodes.columns), searchable=True, label="Column:", value="preds")
    toggle_polygons = mo.ui.switch(value=True, label="Show subsectors")
    return select_node_col, toggle_polygons


@app.cell
def _(network_edges, select_node_col, toggle_polygons, user_nodes):
    _nodes = user_nodes.set_geometry("original_geometry") if toggle_polygons.value else user_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(m=_m, column=select_node_col.value, marker_kwds={"radius": 5}, cmap="OrRd")

    mo.vstack([
        mo.hstack([select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell
def _(network_nodes):
    CRS = "EPSG:4326"
    utm_crs = network_nodes.estimate_utm_crs()
    return (CRS,)


@app.cell
def _():
    import contextily as cx
    import matplotlib.pyplot as plt

    return cx, plt


@app.cell
def _(user_id):
    user_id
    return


@app.cell
def _(CRS, cx):
    from matplotlib.colors import PowerNorm

    def plot_preds(nodes, col, title, ax=None, as_points=False, edges=None):
        nodes = nodes if as_points else nodes.set_geometry("original_geometry")
        nodes = nodes.to_crs(CRS)

        figsize = (7, 7) if ax is None else None

        if not as_points:
            ax = nodes.boundary.plot(ax=ax, color="gray", linewidth=0.3)

        if edges is not None:
            ax = edges.plot(ax=ax, figsize=figsize, color="gray", legend=True, linewidth=0.3)

        cmap = "Reds" if not as_points else "viridis_r"
        gamma = 0.7 if not as_points else 0.3
        label = "Predicted probability of visit $\\hat{y}_{i,n}$" if not as_points else "Per-node visit frequency"

        ax = nodes.plot(
            ax=ax,
            column=col,
            figsize=figsize,
            legend=True,
            cmap=cmap,
            norm=PowerNorm(gamma=gamma, vmin=0, vmax=1),
            legend_kwds={
                "shrink": 0.7,
                "label": label,
                "location": "bottom",
                "pad": 0.01,
            },
            markersize=25,
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    return (plot_preds,)


@app.cell
def _():
    double_figsize = (13, 7)
    return (double_figsize,)


@app.cell
def _(
    double_figsize,
    network_edges,
    plot_preds,
    plot_visited,
    plt,
    user_nodes,
):
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_preds(
        user_nodes,
        "node_preds",
        "Graph structure of the TPG dataset, with per-node visit frequencies",
        ax=_axs[0],
        as_points=True,
        edges=network_edges,
    )

    plot_visited(user_nodes, "is_visited", "Visited nodes for example individual", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_struct.png")
    _fig
    return


@app.cell
def _(double_figsize, plot_preds, plt, user_nodes):
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_preds(user_nodes, "mlp_preds", "MLP predicted visit probabilities for example individual", ax=_axs[0])

    plot_preds(user_nodes, "preds", "GATSkipRes predicted visit probabilities for example individual", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_preds.png")
    _fig
    return


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(CRS, cx, sns):
    from matplotlib.colors import ListedColormap

    def plot_visited(
        nodes, col, title, ax=None, legend_name="Visited", as_points=False, edges=None, include_unvisited=False
    ):
        nodes = nodes if as_points else nodes.set_geometry("original_geometry")
        nodes = nodes.to_crs(CRS).copy()

        if include_unvisited:
            nodes.loc[nodes[col] == 0, "visit"] = "Unvisited"

        nodes.loc[nodes[col] == 1, "visit"] = legend_name
        nodes.loc[nodes["is_home"] == 1, "visit"] = "Home"

        figsize = (7, 7) if ax is None else None

        if not as_points:
            ax = nodes.boundary.plot(ax=ax, color="gray", linewidth=0.3)

        if edges is not None:
            ax = edges.to_crs(CRS).plot(ax=ax, figsize=figsize, color="gray", linewidth=0.3)

        tab10 = sns.color_palette()
        categories = [legend_name, "Home"] + (["Unvisited"] if include_unvisited else [])
        colors = [tab10[0], tab10[1]] + (["lightgray"] if include_unvisited else [])
        cmap = ListedColormap(colors)

        ax = nodes.plot(
            ax=ax,
            column="visit",
            figsize=figsize,
            legend=True,
            markersize=20,
            categories=categories,
            cmap=cmap,
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    return (plot_visited,)


@app.cell
def _(double_figsize, plot_visited, plt, user_nodes):
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_visited(user_nodes, "poisson", "Poisson sample of GATSkipRes predictions", legend_name="Sampled", ax=_axs[0])
    plot_visited(user_nodes, "pps", "15-node PPS sample of GATSkipRes predictions", legend_name="Sampled", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_samples.png")
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
