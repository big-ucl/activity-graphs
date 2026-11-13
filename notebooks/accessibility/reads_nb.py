import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path(mo.notebook_dir())
    project_root = _here.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from notebooks.accessibility import reads


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rail and motorway reads on the saved runs

    Every table is built by `notebooks/accessibility/reads.py`, which prints the same tables for all four
    datasets with `uv run python notebooks/accessibility/reads.py`.

    A zone is *near* a point (a rail or metro station, or a motorway junction) when the point lies within
    1 km of the zone polygon. Two stations share a *line* when one rail or metro trip on the service
    date calls at both. Two junctions share a line when they lie on motorways with a common `ref`.

    **Same-line enrichment.** The test users whose home zone is near a point. In each distance band,
    each ratio is a share among the positives over the same share among the unvisited nodes:

    - `line_ratio`: the share near a point on a line through a point near home;
    - `near_ratio`: the share near any point;
    - `line_given_near_ratio`: the share on a home line among the nodes near any point.

    The second table leaves the 50 zones scored highest by `VisitFrequency` out of both sides.

    **Direction-matched enrichment.** The same users, with the top 50 zones left out. Each positive is
    matched to the unvisited nodes of the same user in its distance band (`_band`), or in its distance
    band and its 45-degree sector of direction from home (`_sector`). Each ratio is the observed count of
    positives on a home line over the count expected from the matched nodes. The drop from `_band` to
    `_sector` is the part of the enrichment explained by direction alone.

    **Far positives.** The positives 9 or more contiguity hops from home, split by whether
    `VisitFrequency` ranks them in its top 50, against the unvisited nodes at the same hops. The
    bounds are 95% bootstrap intervals over users.
    """)
    return


@app.cell
def _():
    dataset_dropdown = mo.ui.dropdown(list(reads.RUNS), value="tts2016", label="Dataset:")
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(dataset_dropdown):
    data = dataset_dropdown.value
    run = reads.read_dataset_run(data)
    networks = [reads.rail_network(run.cfg, run.polygons), reads.motorway_network(data, run.polygons)]

    {"run": reads.RUNS[data], "hop_mismatches": reads.check_positive_hops(run)}
    return networks, run


@app.cell
def _(networks, run):
    network_tabs = {}
    for network in networks:
        summary = pl.DataFrame([reads.network_summary(network, run)])
        sections = [mo.md("Coverage"), mo.ui.table(summary, selection=None)]
        for title, table in reads.network_tables(network, run).items():
            sections.append(mo.md(f"**{title}**"))
            sections.append(mo.ui.table(table, selection=None, page_size=20))
        network_tabs[network.name] = mo.vstack(sections)

    mo.ui.tabs(network_tabs)
    return


if __name__ == "__main__":
    app.run()
