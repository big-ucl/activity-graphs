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
    return cfg, gpd, mo, project_root


@app.cell
def _(cfg, project_root):
    from activitygraphs.data.cmap import load_files

    inputs = load_files(cfg.data, project_root)
    return (inputs,)


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
def _(inputs):
    inputs.boundaries_ct
    return


@app.cell
def _(cfg, gpd, project_root):
    county_fps = [
        "031",  # Cook
        "043",  # DuPage
        "089",  # Kane
        "093",  # Kendall
        "097",  # Lake
        "111",  # McHenry
        "197",  # Will
    ]

    counties = gpd.read_file(
        project_root
        / cfg.data.inputs.boundaries.directory
        / "tl_2025_us_county.zip"
    )
    counties = counties.query("STATEFP == '17'")
    counties
    return (county_fps,)


@app.cell
def _(county_fps, inputs):
    census_tracts = inputs.boundaries_ct[
        inputs.boundaries_ct["COUNTYFP"].isin(county_fps)
    ]
    census_tracts
    return


@app.cell
def _(inputs):
    from activitygraphs.data.cmap import build_cmap_locations

    locations_gdf = build_cmap_locations(inputs)
    locations_gdf
    return (locations_gdf,)


@app.cell
def _(inputs, locations_gdf):
    from activitygraphs.data.cmap import build_cmap_journeys

    user_journeys_df = build_cmap_journeys(inputs, locations_gdf)
    user_journeys_df
    return (user_journeys_df,)


@app.cell
def _(inputs, locations_gdf, user_journeys_df):
    from activitygraphs.data.cmap import build_cmap_users

    build_cmap_users(inputs, locations_gdf, user_journeys_df)
    return


@app.cell
def _(inputs):
    from activitygraphs.data.cmap import build_cmap_data

    cmap_data = build_cmap_data(inputs)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
