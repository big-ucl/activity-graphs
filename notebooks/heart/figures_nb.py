import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo

    from activitygraphs import figures

    # ``mo.notebook_dir`` returns the working directory rather than this file's when the notebook is run
    # headlessly, so the file's own path is preferred wherever it is available.
    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path(mo.notebook_dir())
    project_root = _here.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from notebooks.heart import figures as heart


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Presentation figures

    Every panel is built by `notebooks/heart/figures.py` and written to `reports/figures/heart/`. This
    notebook only drives those builders from its widgets, so a panel looks the same whether it is made
    here or by running that script. `uv run python notebooks/heart/figures.py` writes the whole set at
    the defaults pinned in it.

    Models, the names they are shown under and their colours all come from `figures.DEFAULT_MODELS`,
    so every chart agrees. The imaginary transport network of figure 1 is added in PowerPoint on top of
    the exported panels.
    """)
    return


@app.cell
def _():
    cfg, per_user, score_vectors, home_distances, n_nodes = heart.load_run()
    nodes, network_edges = heart.load_network(cfg)
    out_path = heart.out_dir(cfg)

    {"models": sorted(per_user["name"].unique().to_list()), "n_nodes": n_nodes}
    return cfg, home_distances, n_nodes, network_edges, nodes, out_path, per_user, score_vectors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pick the test user the maps are drawn for. Users with several visited zones make figures 2c and
    2e worth looking at, so the dropdown is ordered by how many they have.
    """)
    return


@app.cell
def _(per_user):
    users = heart.users_by_visits(per_user)
    select_user = mo.ui.dropdown(users, value=users[0], label="User:", searchable=True)
    select_user
    return (select_user,)


@app.cell
def _(per_user, select_user):
    home_node, visited_nodes = heart.user_zones(per_user, select_user.value)
    {"home_node": home_node, "visited_nodes": visited_nodes}
    return home_node, visited_nodes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure 1 - graph structure

    A neighbourhood around the chosen home zone: (a) plain zones, (b) home, work and gym markers,
    (c) the adjacency graph, (d) a subway line crossing the panel through home and a gym,
    (e) the graph and the home zone alone, (f) the graph over the zones with nothing marked. A wider
    ring of zones is drawn than the view shows, so zones fill the panel to every edge, and every panel
    reserves the same strip for its legend, so the frames can be laid over each other.
    """)
    return


@app.cell
def _():
    n_zones = mo.ui.slider(10, 120, value=heart.ZONES_IN_VIEW, label="Zones in view:")
    n_zones
    return (n_zones,)


@app.cell
def _(home_node, n_zones, network_edges, nodes, out_path):
    heart.graph_structure_figures(nodes, network_edges, home_node, out_path, zones_in_view=n_zones.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure 2 - what a ranking looks like

    The region cropped to the home zone and the selection around it: (a) plain, (b) the home zone,
    (c) home and the visited zones, (d) the score choropleth, (e) the top-k selections, one panel per
    cutoff and the whole set side by side. Every panel is drawn at the same geometry and over the same
    view, so the frames can be laid over each other, and the view is flattened to
    `heart.PANEL_MAX_ASPECT`, so a visited zone at the far end of the region falls outside it.

    The choropleth is coloured by `rank_prominence` - `1 - log(rank) / log(n)` - rather than a
    percentile, which would spread colour evenly over the region and flatten the top of the ranking.
    The scale is floored at `figures.MAP_RANK_FLOOR`, so zones ranked below it are drawn white and the
    whole ramp is spent on the ranks a candidate set is drawn from. Its bar is ticked in ranks.
    """)
    return


@app.cell
def _():
    select_map_model = mo.ui.dropdown(
        [spec.name for spec in figures.DEFAULT_MODELS],
        value=heart.MAP_MODEL,
        label="Model:",
    )
    select_seed = mo.ui.dropdown(["none", "42", "43", "44"], value=str(heart.MAP_SEED), label="Seed (none for a baseline):")
    mo.hstack([select_map_model, select_seed], justify="start")
    return select_map_model, select_seed


@app.cell
def _(select_seed):
    seed = None if select_seed.value == "none" else int(select_seed.value)
    return (seed,)


@app.cell
def _(home_node, nodes, out_path, score_vectors, seed, select_map_model, select_user, visited_nodes):
    heart.region_figures(
        nodes,
        score_vectors,
        select_user.value,
        home_node,
        visited_nodes,
        out_path,
        model=select_map_model.value,
        seed=seed,
    )
    return


@app.cell
def _(home_node, n_nodes, nodes, out_path, score_vectors, seed, select_map_model, select_user, visited_nodes):
    heart.ranking_figures(
        nodes,
        score_vectors,
        select_user.value,
        home_node,
        visited_nodes,
        n_nodes,
        out_path,
        model=select_map_model.value,
        seed=seed,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure 3 - spatial vs non-spatial ranking

    The same user under a GNN, a non-spatial model and the factorization baseline, on one colour bar
    ticked in ranks, over the view of figure 2 (e) widened to hold whichever zones any of the three
    selects. (a) holds the first two and (b) adds the third.
    """)
    return


@app.cell
def _():
    _names = [spec.name for spec in figures.DEFAULT_MODELS]
    select_spatial = mo.ui.dropdown(_names, value=heart.SPATIAL_MODEL, label="Spatial model:")
    select_flat = mo.ui.dropdown(_names, value=heart.FLAT_MODEL, label="Non-spatial model:")
    select_mf = mo.ui.dropdown(_names, value=heart.MF_MODEL, label="Factorization model:")
    mo.hstack([select_spatial, select_flat, select_mf], justify="start")
    return select_flat, select_mf, select_spatial


@app.cell
def _(
    home_node,
    n_nodes,
    nodes,
    out_path,
    score_vectors,
    seed,
    select_flat,
    select_mf,
    select_spatial,
    select_user,
    visited_nodes,
):
    heart.spatial_vs_flat_figure(
        nodes,
        score_vectors,
        select_user.value,
        home_node,
        visited_nodes,
        n_nodes,
        out_path,
        spatial=select_spatial.value,
        flat=select_flat.value,
        factorization=select_mf.value,
        seed=seed,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure 4 - recall@k curve

    Then the area under each model's recall@k curve, read against the share of the region selected,
    and `avg_recall@K`.
    """)
    return


@app.cell
def _(cfg, n_nodes, out_path, per_user):
    auc_table, avg_recall = heart.recall_figures(cfg, per_user, n_nodes, out_path)
    mo.vstack([auc_table, avg_recall])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure 5 - performance by distance from home

    Mean recall of the visited locations in each distance band from home, the figure of the progress
    report. Each band's axis label carries its positives and their share of all of them.
    """)
    return


@app.cell
def _(cfg, home_distances, out_path, per_user):
    band_table = heart.band_figures(cfg, per_user, home_distances, out_path)
    band_table
    return


if __name__ == "__main__":
    app.run()
