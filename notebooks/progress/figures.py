"""Build the figures of the progress chapter into ``reports/figures/progress/``.

Run with ``uv run python notebooks/progress/figures.py``. Every figure is read off the parquets of
one run per dataset, pinned in ``RUNS``, so nothing here re-trains a model.

Hue carries the architecture and the dash pattern carries the input features, which is what keeps
the eight lines of the recall curve apart without eight hues.
"""

from pathlib import Path

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl

from activitygraphs import analysis, figures
from activitygraphs.config import load_config
from activitygraphs.figures import DASHED, DOTTED, SOLID, ModelStyle
from activitygraphs.ml.training import PER_USER_STAGE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports/figures/progress"

RUNS = {"cmap": ("CMAP", 9), "geneva": ("GenevaTPG", 5)}
PANEL_TITLES = {"cmap": "CMAP (Chicago)", "geneva": "Geneva"}

GNN = "GATSkip-8-res-hpe-dist"
MLP = "MLP-dist"
REFERENCE = "Gravity"

# The five variants of the architecture table plus the three baselines the chapter reads them against.
CURVE_MODELS: tuple[ModelStyle, ...] = (
    ModelStyle(GNN, "GNN + home PE + dist", "#d1242f", learned=True, dash=SOLID),
    ModelStyle("GATSkip-8-res-dist", "GNN + dist", "#d1242f", learned=True, dash=DASHED),
    ModelStyle("GATSkip-8-res", "GNN", "#d1242f", learned=True, dash=DOTTED),
    ModelStyle(MLP, "MLP + dist", "#1f6feb", learned=True, dash=SOLID),
    ModelStyle("MLP", "MLP", "#1f6feb", learned=True, dash=DOTTED),
    ModelStyle(REFERENCE, "Gravity", "#bf8700", learned=False, dash=SOLID),
    ModelStyle("VisitFrequency", "VisitFrequency", "#57606a", learned=False, dash=DASHED),
    ModelStyle("Uniform", "Uniform", "#8c959f", learned=False, dash=DOTTED),
)

# The distance-band figure carries the two architectures and the baselines that trade near visits for far ones.
BAND_MODELS: tuple[ModelStyle, ...] = (
    CURVE_MODELS[0],
    CURVE_MODELS[3],
    CURVE_MODELS[5],
    CURVE_MODELS[6],
    CURVE_MODELS[7],
)

PANEL = (620, 260)

# The zone maps draw web tiles underneath, which needs network access.
BASEMAP = True

# The map palette and the rank floor live in ``activitygraphs.figures`` so this chapter's maps and the hEART
# ones cannot drift apart. Rebinding them here keeps the call sites below readable.
MAP_RANK_FLOOR = figures.MAP_RANK_FLOOR
MAP_RANK_TICKS = figures.MAP_RANK_TICKS
MAP_BELOW_FLOOR = figures.MAP_BELOW_FLOOR
MAP_CMAP = figures.MAP_CMAP

HOME_COLOUR, VISITED_COLOUR = figures.ZONE_COLOURS[0], figures.ZONE_COLOURS[1]
MARKER_EDGE = "none"
SELECTION_COLOUR = figures.HIGHLIGHT_COLOUR
HIGHLIGHT_WIDTH = 0.8


def load(dataset: str):
    """The config, per-user results, score vectors and home distances of a dataset's pinned run."""
    name, run = RUNS[dataset]
    cfg = load_config(PROJECT_ROOT, verbose=False, data=dataset)
    reports = PROJECT_ROOT / cfg.paths.reports_data

    _, per_user = analysis.load_report_run(reports, name, run, cfg.train.max_recall_k)
    scores = analysis.load_score_vectors(reports, name, run)
    distances = analysis.load_home_distances(reports, name, run)

    return cfg, per_user, scores, distances


def save_chart(chart, stem: str) -> None:
    """Write an Altair chart as a vector PDF and a raster PNG beside it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / f"{stem}.pdf")
    chart.save(OUT_DIR / f"{stem}.png", scale_factor=3)
    print(f"  wrote {stem}.pdf / .png")


def save_figure(fig, stem: str) -> None:
    """Write a matplotlib figure as a vector PDF and a raster PNG beside it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  wrote {stem}.pdf / .png")


# =========================================
# Recall@k curve
# =========================================


def recall_curve_figure() -> None:
    """The recall@k curve of every variant and the three reference baselines, one panel per dataset."""
    panels = []

    for dataset in RUNS:
        cfg, per_user, scores, _ = load(dataset)
        n_nodes = len(scores["scores"].to_list()[0])
        budget = cfg.train.max_recall_k
        ks = list(range(1, budget + 1))

        curve = analysis.recall_curve(per_user, ks)
        print(f"  {dataset}: {n_nodes} nodes, k = {budget} is {budget / n_nodes:.1%} of them")
        panels.append(
            figures.recall_chart(
                curve,
                CURVE_MODELS,
                size=PANEL,
                tick_ks=[1, 10, 20, 30, 40, 50],
                points=False,
                title=PANEL_TITLES[dataset],
                style=False,
                log_k=False,
            )
        )

    save_chart(stacked(panels), "progress-recall-curve")


def stacked(panels: list, shared_y: bool = True) -> alt.VConcatChart:
    """The dataset panels one above the other under one shared legend, styled as one chart.

    The value scale is shared so the panels can be read against one another, which is the whole point of
    putting the two datasets in one figure.
    """
    concatenated = alt.vconcat(*panels, spacing=30).resolve_scale(
        color="shared", strokeDash="shared", y="shared" if shared_y else "independent"
    )

    return figures.chart_style(concatenated)


# =========================================
# Performance by distance from home
# =========================================


def distance_band_figure() -> None:
    """Mean recall of the visited locations in each distance band from home, one panel per dataset."""
    panels = []
    keep = [spec.name for spec in BAND_MODELS]

    for dataset in RUNS:
        cfg, per_user, _, distances = load(dataset)

        band = analysis.distance_band_recall(
            per_user.filter(pl.col("name").is_in(keep)), distances, list(cfg.data.distance_bands)
        )
        panels.append(
            figures.band_recall_chart(
                analysis.distance_band_recall_table(band),
                cfg.train.max_recall_k,
                BAND_MODELS,
                size=PANEL,
                title=PANEL_TITLES[dataset],
                style=False,
            )
        )

    save_chart(stacked(panels), "progress-distance-bands")


# =========================================
# One individual's ranking on the map
# =========================================


def pick_user(per_user: pl.DataFrame, min_positives: int = 4) -> int:
    """The lowest-numbered test user with at least ``min_positives`` visited nodes besides home."""
    rows = per_user.filter(
        (pl.col("stage") == PER_USER_STAGE)
        & (pl.col("name") == REFERENCE)
        & (pl.col("pos_node").list.len() >= min_positives)
    ).sort("user_id")

    if rows.is_empty():
        raise ValueError(f"no test user has {min_positives} positives")

    return int(rows["user_id"][0])


def example_ranking_figure(k: int = 20, seed: int = 42, user_id: int | None = None) -> None:
    """One CMAP individual's scores under the GNN and the node-level MLP, with the top-k set outlined.

    Both panels are coloured by rank prominence rather than by the raw score, so the two models sit on one
    colour bar although their scores are on different scales.
    """
    cfg, per_user, scores, _ = load("cmap")
    nodes = figures.node_gdf(gpd.read_parquet(PROJECT_ROOT / cfg.data.paths.processed / "NetworkGraph/nodes.parquet"))

    user_id = pick_user(per_user) if user_id is None else user_id
    row = per_user.filter((pl.col("stage") == PER_USER_STAGE) & (pl.col("user_id") == user_id)).row(0, named=True)
    home_node, visited = int(row["home_node"]), [int(node) for node in row["pos_node"]]

    by_model = {
        spec.label: figures.rank_prominence(figures.user_scores(scores, spec.name, user_id, seed))
        for spec in (CURVE_MODELS[0], CURVE_MODELS[3])
    }
    top_k = {label: figures.top_k_nodes(values, k) for label, values in by_model.items()}
    focus = sorted({home_node, *visited, *(int(node) for selected in top_k.values() for node in selected)})

    fig, axes = plt.subplots(1, len(by_model), figsize=(7.5, 7.5), gridspec_kw={"wspace": 0.03})

    for ax, (label, values) in zip(axes, by_model.items(), strict=True):
        figures.zone_map(
            nodes,
            values=values,
            highlight_zones={"Home": [home_node], "Visited": visited},
            highlight=top_k[label],
            focus=focus,
            ax=ax,
            legend=False,
            basemap=BASEMAP,
            boundary_width=0.25,
            cmap=MAP_CMAP,
            **figures.rank_limits(len(nodes), MAP_RANK_FLOOR),
            zone_colours=[HOME_COLOUR, VISITED_COLOUR],
            zone_edge=MARKER_EDGE,
            highlight_colour=SELECTION_COLOUR,
            highlight_width=HIGHLIGHT_WIDTH,
            title=label,
        )

    add_map_key(fig, axes, k, len(nodes))
    print(f"  user {user_id}: home node {home_node}, {len(visited)} visited nodes")
    save_figure(fig, "progress-example-ranking")


def add_map_key(fig, axes, k: int, n_nodes: int) -> None:
    """One colour bar and one marker legend for both panels, below them rather than between them."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figures.rank_colour_bar(fig, list(axes), n_nodes, floor=MAP_RANK_FLOOR, ticks=MAP_RANK_TICKS)

    handles = [
        Patch(facecolor=HOME_COLOUR, edgecolor=MARKER_EDGE, label="Home zone"),
        Patch(facecolor=VISITED_COLOUR, edgecolor=MARKER_EDGE, label="Visited zone"),
        Line2D([], [], color=SELECTION_COLOUR, linewidth=HIGHLIGHT_WIDTH, label=f"Top {k} selected"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=10)


def main() -> None:
    print("recall@k curve:")
    recall_curve_figure()
    print("performance by distance from home:")
    distance_band_figure()
    print("one individual's ranking:")
    example_ranking_figure()


if __name__ == "__main__":
    main()
