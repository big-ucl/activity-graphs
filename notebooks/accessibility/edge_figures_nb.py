import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo

    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path(mo.notebook_dir())
    project_root = _here.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from notebooks.accessibility import edge_figures as ef


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 6 figures: the band reads and the rail edges on a map

    Every chart is built by `notebooks/accessibility/edge_figures.py`.

    **Heatmaps.** One row per model, one column per band from home. The colour is the paired difference against the
    reference model, per user: blue where the model does better, red where it does worse, grey near zero. A dot marks
    a difference whose 95% bootstrap interval over users excludes zero. Hover a cell for the interval.

    **Maps.** The mean paired change in recall between two models in every cell: a zone, or a hexagon pooling the
    zones whose centre it holds. By *destination*, the mean change in the per-positive `avg_recall` of the test
    positives in the cell. By *home*, the mean change in the per-user `avg_recall` of the test users living in it.
    Each model is averaged over its seeds first. Cells with fewer positives or users than the threshold are grey. A
    dark outline marks a cell whose 95% interval excludes zero; the interval treats the positives of one user as
    independent, so it is a little narrow. Rings mark the station zones, sized by rail degree, the number of zones a
    rail edge joins the zone to. The scatter plots every shown cell against the highest rail degree in it.

    The first load of a run computes its per-user within-band AUC, a few minutes on TTS, and caches it.
    """)
    return


@app.cell
def _():
    dataset_dropdown = mo.ui.dropdown(list(ef.RUNS), value="tts2016", label="Dataset:")
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(dataset_dropdown):
    with mo.status.spinner(title=f"Loading {dataset_dropdown.value} run {ef.RUNS[dataset_dropdown.value]}"):
        reads = ef.load_run_reads(dataset_dropdown.value, ef.RUNS[dataset_dropdown.value])
    trained_names = [name for name in reads.names if name.startswith(("MLP", "GATv2", "TwoTower", "GTransformer"))]
    norail_names = [name for name in reads.names if name.endswith("-norail") and "nocontig" not in name]
    return norail_names, reads, trained_names


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The band reads
    """)
    return


@app.cell
def _(norail_names, reads, trained_names):
    reference_dropdown = mo.ui.dropdown(
        reads.names, value=norail_names[0] if norail_names else "Gravity", label="Reference:"
    )
    metric_dropdown = mo.ui.dropdown(list(ef.METRICS), value=ef.METRICS[0], label="Read:")
    models_select = mo.ui.multiselect(reads.names, value=trained_names, label="Models:")
    mo.hstack([reference_dropdown, metric_dropdown, models_select], justify="start")
    return metric_dropdown, models_select, reference_dropdown


@app.cell
def _(metric_dropdown, models_select, reads, reference_dropdown):
    _reference = reference_dropdown.value
    _models = [name for name in models_select.value if name != _reference]
    with mo.status.spinner(title="Pairing every model with the reference"):
        _differences = ef.band_differences(reads, _reference, _models, metric_dropdown.value)
    ef.band_heatmap(_differences, _reference, metric_dropdown.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The change between two models, by zone
    """)
    return


@app.cell
def _(reads, trained_names):
    _default_model = "GATv2-2" if "GATv2-2" in reads.names else trained_names[0]
    _default_reference = f"{_default_model}-norail" if f"{_default_model}-norail" in reads.names else "Gravity"
    model_dropdown = mo.ui.dropdown(reads.names, value=_default_model, label="Model:")
    map_reference_dropdown = mo.ui.dropdown(reads.names, value=_default_reference, label="Minus:")
    by_radio = mo.ui.radio(["destination", "home"], value="destination", label="By zone of:", inline=True)
    cells_radio = mo.ui.radio(["zones", "hexagons"], value="hexagons", label="Cells:", inline=True)
    hex_km_slider = mo.ui.slider(1, 20, value=ef.DEFAULT_HEX_KM, step=0.5, label="Hexagon width (km):")
    min_count_slider = mo.ui.slider(1, 500, value=ef.DEFAULT_MIN_ZONE_COUNT, label="Fewest to show:")
    mo.vstack([
        mo.hstack([model_dropdown, map_reference_dropdown, by_radio], justify="start"),
        mo.hstack([cells_radio, hex_km_slider, min_count_slider], justify="start"),
    ])
    return by_radio, cells_radio, hex_km_slider, map_reference_dropdown, min_count_slider, model_dropdown


@app.cell
def _(by_radio, cells_radio, hex_km_slider, map_reference_dropdown, min_count_slider, model_dropdown, reads):
    _zones = ef.zone_changes(
        reads, model_dropdown.value, map_reference_dropdown.value, by_radio.value, min_count_slider.value
    )
    if cells_radio.value == "hexagons":
        cell_changes = ef.hex_changes(_zones, hex_km_slider.value, min_count_slider.value)
    else:
        cell_changes = _zones
    _stations = reads.zones[reads.zones["rail_degree"] > 0]
    ef.zone_map(cell_changes, _stations, model_dropdown.value, map_reference_dropdown.value, by_radio.value)
    return (cell_changes,)


@app.cell
def _(cell_changes, map_reference_dropdown, model_dropdown):
    ef.degree_scatter(cell_changes, model_dropdown.value, map_reference_dropdown.value)
    return


if __name__ == "__main__":
    app.run()
