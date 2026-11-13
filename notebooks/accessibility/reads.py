"""Read the saved runs against the rail and motorway networks: same-line enrichment and far positives.

Run with ``uv run python notebooks/accessibility/reads.py`` to print every table for every dataset. The marimo
notebook beside this file, ``reads_nb.py``, imports these functions and shows the same tables.

A zone is near a network point (a rail or metro station, or a motorway junction) when the point lies within
``RADIUS_M`` of the zone polygon. Two rail stations share a line when one rail or metro trip on the service date of
the dataset calls at both. Two motorway junctions share a line when they lie on motorways with a common ``ref``.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from activitygraphs.analysis import POPULARITY_MODEL, read_home_distances
from activitygraphs.base import PER_USER_STAGE
from activitygraphs.config import Config, read_config
from activitygraphs.runs import run_dir
from activitygraphs.surveys.gtfs import PTMode, read_gtfs
from activitygraphs.training.metrics import DEFAULT_HOP_BANDS
from activitygraphs.training.ranking import expected_recall_at_k

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Run read for each dataset.
RUNS = {"tpg": 3, "cmap": 17, "tts2016": 11, "tts2022": 3}

# OSM extract of each dataset, clipped to the zone hull plus 10 km.
OSM_CLIPS = {
    "tpg": "tpg-220101-clipped.osm.pbf",
    "cmap": "cmap-190101-clipped.osm.pbf",
    "tts2016": "tts2016-170101-clipped.osm.pbf",
    "tts2022": "tts2022-230101-clipped.osm.pbf",
}

RADIUS_M = 1000.0
RAIL_MODES = [PTMode.RAIL.value, PTMode.METRO.value]
TOP_K = 50
FAR_HOPS = 9
N_BOOTSTRAP = 2000
N_SECTORS = 8

# Columns of the per-user counts of the same-line read, in order.
ENRICHMENT_COUNTS = ["pos_n", "pos_line", "pos_near", "unv_n", "unv_line", "unv_near"]
# Columns of the per-user counts of the direction-matched read, in order: the positives in a stratum with unvisited
# nodes, then the observed and expected positives on a home line, among all nodes and among the nodes near a point.
MATCHED_COUNTS = ["pos_n", "obs_line", "exp_line", "obs_line_near", "exp_line_near"]
# Groups of the far-positive read: the positives inside and outside the popularity top k, and the unvisited nodes.
FAR_GROUPS = [f"pos_top{TOP_K}", f"pos_beyond{TOP_K}", "unvisited"]


@dataclass(frozen=True)
class Network:
    """Points of one transport network near the zones, and the lines joining them.

    Attributes:
        name: Name of the network, e.g. ``"rail"``.
        near_zone: ``[N, P]`` sparse, 1 where point ``p`` lies within ``RADIUS_M`` of zone ``n``.
        same_line: ``[P, P]`` sparse, 1 where points ``p`` and ``q`` share a line, the diagonal included.
    """

    name: str
    near_zone: sp.csr_matrix
    same_line: sp.csr_matrix

    @property
    def served_zones(self) -> np.ndarray:
        """``[N]`` bool, true for the zones near at least one point."""
        return np.asarray(self.near_zone.sum(axis=1)).ravel() > 0

    @property
    def home_line_zones(self) -> np.ndarray:
        """``[N, N]`` bool, true at ``[h, n]`` where zone ``n`` is near a point on a line through a point near ``h``."""
        line_reach = self.near_zone @ self.same_line @ self.near_zone.T
        return line_reach.toarray() > 0


@dataclass(frozen=True)
class DatasetRun:
    """Zones, contiguity hops and test users of one dataset, read off one saved run.

    Attributes:
        cfg: Config of the dataset.
        polygons: Zone polygons in node order, in the UTM zone of the dataset.
        centroids: ``[N, 2]`` zone points in node order, in metres in the UTM zone of the dataset, the points from
            which ``distance_m`` is measured.
        hops: ``[N, N]`` contiguity-hop distance between every pair of nodes.
        home_distance: ``[N, N]`` distance in metres from each home zone of a test user to every node. Rows of zones
            that are no test user's home are NaN.
        positives: One row per positive of a test user: ``user, home_node, pos_node, pos_hops, pos_top_k``.
            ``pos_top_k`` is the fraction of the ranks of the positive under the popularity baseline that are
            ``<= TOP_K``.
        home_nodes: ``[U]`` home node of each test user, in ``user`` order.
        popular_zones: ``[N]`` bool, true for the ``TOP_K`` zones scored highest by the popularity baseline.
    """

    cfg: Config
    polygons: gpd.GeoSeries
    centroids: np.ndarray
    hops: np.ndarray
    home_distance: np.ndarray
    positives: pl.DataFrame
    home_nodes: np.ndarray
    popular_zones: np.ndarray


def read_dataset_run(data: str) -> DatasetRun:
    """Read the zones, the contiguity hops and the test users of ``data`` from its graph cache and its run.

    The run is the one numbered in ``RUNS``.
    """
    cfg = read_config(PROJECT_ROOT, data=data)
    graph_path = PROJECT_ROOT / cfg.data.paths.processed / "Graph"

    nodes = gpd.read_parquet(graph_path / "nodes.parquet").sort_index()
    utm_crs = nodes.estimate_utm_crs()
    centroids = nodes.to_crs(utm_crs).geometry.get_coordinates().to_numpy()
    polygons = nodes.set_geometry("original_geometry").to_crs(utm_crs).geometry.reset_index(drop=True)
    edges = gpd.read_parquet(graph_path / "edges.parquet")
    hops = _contiguity_hops(nodes.index.to_list(), edges)

    run_path = _run_path(cfg, data)
    per_user = _popularity_rows(run_path)
    user_homes = per_user.select("user", "run_user", "home_node").sort("user")

    distances = read_home_distances(run_path).rename({"user_idx": "run_user"})
    home_distance = _home_distance_matrix(user_homes, distances, len(polygons))

    ranked_positives = per_user.select(
        "user", "home_node", "pos_node", "pos_hops", "pos_n_scored_higher", "pos_n_tied"
    ).explode("pos_node", "pos_hops", "pos_n_scored_higher", "pos_n_tied")
    n_higher = ranked_positives["pos_n_scored_higher"].to_numpy().astype(np.float64)
    n_tied = ranked_positives["pos_n_tied"].to_numpy().astype(np.float64)
    top_k_fraction = expected_recall_at_k(n_higher, n_tied, TOP_K)
    positives = ranked_positives.select("user", "home_node", "pos_node", "pos_hops").with_columns(
        pos_top_k=top_k_fraction
    )

    return DatasetRun(
        cfg=cfg,
        polygons=polygons,
        centroids=centroids,
        hops=hops,
        home_distance=home_distance,
        positives=positives,
        home_nodes=user_homes["home_node"].to_numpy(),
        popular_zones=_popular_zones(run_path),
    )


def rail_network(cfg: Config, polygons: gpd.GeoSeries) -> Network:
    """Return the rail and metro stations served on the service date of ``cfg``, and the trips joining them.

    A stop counts as its parent station where the feed gives one.
    """
    gtfs = read_gtfs(cfg.data.inputs.gtfs, PROJECT_ROOT / cfg.data.paths.gtfs)

    rail_routes = gtfs.routes_df.filter(pl.col("mode").is_in(RAIL_MODES)).select("route_id")
    rail_trips = gtfs.trips_on_service_date().join(rail_routes.lazy(), on="route_id").select("trip_id")
    stop_stations = gtfs.stops_df.select("stop_id", point_id=pl.coalesce("parent_station", "stop_id"))
    station_calls = (
        gtfs.stop_times_df
        .join(rail_trips, on="trip_id")
        .select("trip_id", "stop_id")
        .collect()
        .join(stop_stations, on="stop_id")
        .select("point_id", line_id="trip_id")
        .unique()
    )

    station_coords = gtfs.stops_df.select(point_id="stop_id", lon="stop_lon", lat="stop_lat")
    stations = station_calls.select("point_id").unique().join(station_coords, on="point_id").sort("point_id")

    return _build_network("rail", stations, station_calls, polygons)


def motorway_network(data: str, polygons: gpd.GeoSeries) -> Network:
    """Return the motorway junctions of the OSM clip of ``data``, joined by the ``ref`` of the motorways they lie on.

    A ``ref`` listing several routes (``"I 90;I 94"``) puts the junction on each of them.
    """
    clip_path = PROJECT_ROOT / "data/external/osm" / OSM_CLIPS[data]
    features = _motorway_features(clip_path).to_crs(polygons.crs)

    is_junction = features["highway"] == "motorway_junction"
    junctions = features[is_junction].reset_index(drop=True)
    junctions["point_id"] = junctions.index.astype(str)
    motorways = features[(features["highway"] == "motorway") & features["ref"].notna()]

    junction_ways = junctions[["point_id", "geometry"]].sjoin(
        motorways[["ref", "geometry"]], predicate="dwithin", distance=1.0
    )
    junction_refs = (
        pl
        .from_pandas(junction_ways[["point_id", "ref"]])
        .with_columns(line_id=pl.col("ref").str.split(";"))
        .explode("line_id")
        .select("point_id", line_id=pl.col("line_id").str.replace_all(r"[\s-]", "").str.to_uppercase())
        .unique()
    )

    junction_coords = junctions.to_crs(4326).get_coordinates()
    points = pl.DataFrame({
        "point_id": junctions["point_id"].to_list(),
        "lon": junction_coords["x"].to_numpy(),
        "lat": junction_coords["y"].to_numpy(),
    })

    return _build_network("motorway", points, junction_refs, polygons)


def same_line_enrichment(
    network: Network, run: DatasetRun, excluded_zones: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> pl.DataFrame:
    """Compare, per distance band, how often positives and unvisited nodes lie near the line through the home zone.

    The test users read are those whose home zone is near a point of ``network``. In each band of
    ``cfg.data.distance_bands`` three ratios are reported, each a share among the positives over the same share among
    the unvisited nodes, pooled over users: ``line_ratio``, the share near a point on a home line; ``near_ratio``, the
    share near any point; and ``line_given_near_ratio``, the share near a point on a home line among the nodes near
    any point.

    Args:
        network: The rail or motorway network.
        run: The dataset run.
        excluded_zones: ``[N]`` bool, zones left out of both the positives and the unvisited nodes.
        n_bootstrap: Number of bootstrap resamples over users.

    Returns:
        Frame of ``band, n_users, n_pos, pos_line_share, unv_line_share``, then each ratio with its ``_lo`` and ``_hi``
        bootstrap bounds over users (95%). ``n_users`` counts the users with a positive in the band.
    """
    served_zones = network.served_zones
    home_line = network.home_line_zones
    bands = run.cfg.data.distance_bands
    n_bands = len(bands) + 1

    user_rows, home_nodes, row_of_user = _user_rows(served_zones[run.home_nodes], run.home_nodes)
    n_users = len(home_nodes)

    node_bands = _distance_bands(run)
    home_counts = _home_stratum_counts(node_bands, n_bands, home_line, served_zones, ~excluded_zones)

    kept_nodes = np.flatnonzero(~excluded_zones).tolist()
    positives = run.positives.filter(pl.col("user").is_in(user_rows.tolist()) & pl.col("pos_node").is_in(kept_nodes))
    pos_rows = row_of_user[positives["user"].to_numpy()]
    pos_homes = positives["home_node"].to_numpy()
    pos_nodes = positives["pos_node"].to_numpy()
    pos_bands = node_bands[pos_homes, pos_nodes]
    pos_flags = np.stack([np.ones(len(pos_nodes)), home_line[pos_homes, pos_nodes], served_zones[pos_nodes]], axis=-1)

    pos_counts = np.zeros((n_users, n_bands, 3))
    np.add.at(pos_counts, (pos_rows, pos_bands), pos_flags)
    unv_counts = home_counts[home_nodes] - pos_counts
    user_counts = np.concatenate([pos_counts, unv_counts], axis=-1)

    count_sums = user_counts.sum(axis=0)
    boot_sums = _bootstrap_sums(user_counts.reshape(n_users, -1), n_bootstrap).reshape(n_bootstrap, n_bands, -1)

    count_index = {count_name: idx for idx, count_name in enumerate(ENRICHMENT_COUNTS)}
    pos_n, pos_line = count_sums[:, count_index["pos_n"]], count_sums[:, count_index["pos_line"]]
    unv_n, unv_line = count_sums[:, count_index["unv_n"]], count_sums[:, count_index["unv_line"]]
    users_with_pos = user_counts[..., count_index["pos_n"]] > 0

    columns: dict[str, object] = {
        "band": _band_labels(bands),
        "n_users": users_with_pos.sum(axis=0),
        "n_pos": pos_n.astype(np.int64),
        "pos_line_share": pos_line / pos_n,
        "unv_line_share": unv_line / unv_n,
    }
    ratio_terms = {
        "line_ratio": ("pos_line", "pos_n", "unv_line", "unv_n"),
        "near_ratio": ("pos_near", "pos_n", "unv_near", "unv_n"),
        "line_given_near_ratio": ("pos_line", "pos_near", "unv_line", "unv_near"),
    }
    for ratio_name, count_names in ratio_terms.items():
        pos_num, pos_den, unv_num, unv_den = (count_index[count_name] for count_name in count_names)
        ratio = _share_ratio(count_sums, pos_num, pos_den, unv_num, unv_den)
        boot_ratio = _share_ratio(boot_sums, pos_num, pos_den, unv_num, unv_den)
        columns[ratio_name] = ratio
        columns[f"{ratio_name}_lo"] = np.nanquantile(boot_ratio, 0.025, axis=0)
        columns[f"{ratio_name}_hi"] = np.nanquantile(boot_ratio, 0.975, axis=0)

    return pl.DataFrame(columns)


def direction_matched_enrichment(
    network: Network, run: DatasetRun, excluded_zones: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> pl.DataFrame:
    """Compare the positives on a home line with the number expected from the unvisited nodes matched to them.

    The test users read are those whose home zone is near a point of ``network``. Each positive is matched to the
    unvisited nodes of the same user, first in its distance band (``band``), then in its distance band and its sector of
    ``N_SECTORS`` equal directions from home (``sector``). The expected count of a positive is the share of its matched
    nodes that lie on a home line. A positive with no matched node is left out. Each ratio is the observed count over
    the expected count, summed over users: ``line``, among all nodes, and ``line_given_near``, among the nodes near any
    point.

    Args:
        network: The rail or motorway network.
        run: The dataset run.
        excluded_zones: ``[N]`` bool, zones left out of both the positives and the unvisited nodes.
        n_bootstrap: Number of bootstrap resamples over users.

    Returns:
        Frame of ``band, n_pos_band, n_pos_sector``, then ``line_band, line_sector, line_given_near_band,
        line_given_near_sector``, each with its ``_lo`` and ``_hi`` bootstrap bounds over users (95%).
        ``n_pos_*`` counts the positives with a matched node.
    """
    served_zones = network.served_zones
    home_line = network.home_line_zones
    bands = run.cfg.data.distance_bands
    n_bands = len(bands) + 1

    user_rows, home_nodes, row_of_user = _user_rows(served_zones[run.home_nodes], run.home_nodes)
    n_users = len(home_nodes)

    kept_nodes = np.flatnonzero(~excluded_zones).tolist()
    positives = run.positives.filter(pl.col("user").is_in(user_rows.tolist()) & pl.col("pos_node").is_in(kept_nodes))
    pos_rows = row_of_user[positives["user"].to_numpy()]
    pos_homes = positives["home_node"].to_numpy()
    pos_nodes = positives["pos_node"].to_numpy()
    pos_flags = np.stack([np.ones(len(pos_nodes)), home_line[pos_homes, pos_nodes], served_zones[pos_nodes]], axis=-1)

    node_bands = _distance_bands(run)
    node_sectors = _direction_sectors(run.centroids)
    matchings = {"band": (node_bands, 1), "sector": (node_bands * N_SECTORS + node_sectors, N_SECTORS)}

    count_index = {count_name: idx for idx, count_name in enumerate(MATCHED_COUNTS)}
    columns: dict[str, object] = {"band": _band_labels(bands)}
    ratio_columns: dict[str, object] = {}
    for matching_name, (node_strata, n_sectors) in matchings.items():
        n_strata = n_bands * n_sectors
        home_counts = _home_stratum_counts(node_strata, n_strata, home_line, served_zones, ~excluded_zones)

        pos_counts = np.zeros((n_users, n_strata, 3))
        pos_strata = node_strata[pos_homes, pos_nodes]
        np.add.at(pos_counts, (pos_rows, pos_strata), pos_flags)
        unv_counts = home_counts[home_nodes] - pos_counts

        matched_counts = _matched_counts(pos_counts, unv_counts)
        user_counts = matched_counts.reshape(n_users, n_bands, n_sectors, -1).sum(axis=2)
        count_sums = user_counts.sum(axis=0)
        boot_sums = _bootstrap_sums(user_counts.reshape(n_users, -1), n_bootstrap).reshape(n_bootstrap, n_bands, -1)

        columns[f"n_pos_{matching_name}"] = count_sums[:, count_index["pos_n"]].astype(np.int64)
        for ratio_name, (obs_name, exp_name) in {
            "line": ("obs_line", "exp_line"),
            "line_given_near": ("obs_line_near", "exp_line_near"),
        }.items():
            obs_idx, exp_idx = count_index[obs_name], count_index[exp_name]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = count_sums[:, obs_idx] / count_sums[:, exp_idx]
                boot_ratio = boot_sums[..., obs_idx] / boot_sums[..., exp_idx]
            column_name = f"{ratio_name}_{matching_name}"
            ratio_columns[column_name] = ratio
            ratio_columns[f"{column_name}_lo"] = np.nanquantile(boot_ratio, 0.025, axis=0)
            ratio_columns[f"{column_name}_hi"] = np.nanquantile(boot_ratio, 0.975, axis=0)

    ratio_order = ["line_band", "line_sector", "line_given_near_band", "line_given_near_sector"]
    for column_name in ratio_order:
        for suffix in ("", "_lo", "_hi"):
            columns[f"{column_name}{suffix}"] = ratio_columns[f"{column_name}{suffix}"]

    return pl.DataFrame(columns)


def far_positive_shares(
    zone_flags: np.ndarray, run: DatasetRun, user_mask: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> pl.DataFrame:
    """Compare the share of flagged zones among the far positives, inside and beyond the popularity top k.

    A far node lies ``FAR_HOPS`` or more contiguity hops from home. The positives are split by their top-k fraction
    under the popularity baseline, so a positive tied across the cutoff counts in both groups.

    Args:
        zone_flags: ``[N, N]`` bool, true at ``[h, n]`` where zone ``n`` is flagged for a user with home ``h``.
        run: The dataset run.
        user_mask: ``[U]`` bool, the test users read.
        n_bootstrap: Number of bootstrap resamples over users.

    Returns:
        Frame of ``hop_band, group, n, share, share_lo, share_hi``: ``n`` is the number of nodes in the group, summed
        over users, and ``share`` the flagged share of them, with bootstrap bounds over users (95%).
    """
    far_bands = [(label, low, high) for label, low, high in DEFAULT_HOP_BANDS if low >= FAR_HOPS]
    far_bands.append((f"{FAR_HOPS}+", FAR_HOPS, float("inf")))
    n_bands = len(far_bands)

    user_rows, home_nodes, row_of_user = _user_rows(user_mask, run.home_nodes)
    n_users = len(home_nodes)

    positives = run.positives.filter(pl.col("user").is_in(user_rows.tolist()))
    pos_rows = row_of_user[positives["user"].to_numpy()]
    pos_homes = positives["home_node"].to_numpy()
    pos_nodes = positives["pos_node"].to_numpy()
    pos_hops = positives["pos_hops"].to_numpy()
    pos_top_k = positives["pos_top_k"].to_numpy()
    pos_flags = zone_flags[pos_homes, pos_nodes]

    # counts[u, band, group, (n, flagged)]
    user_counts = np.zeros((n_users, n_bands, len(FAR_GROUPS), 2))
    for band_idx, (_, low, high) in enumerate(far_bands):
        in_band = (pos_hops >= low) & (pos_hops <= high)
        for group_idx, group_weight in enumerate([pos_top_k, 1.0 - pos_top_k]):
            weights = np.where(in_band, group_weight, 0.0)
            np.add.at(user_counts[:, band_idx, group_idx, 0], pos_rows, weights)
            np.add.at(user_counts[:, band_idx, group_idx, 1], pos_rows, weights * pos_flags)

        in_home_band = (run.hops >= low) & (run.hops <= high)
        band_counts = in_home_band.sum(axis=1)
        flagged_counts = (in_home_band & zone_flags).sum(axis=1)
        pos_in_band = np.where(in_band, 1.0, 0.0)
        pos_in_band_flagged = pos_in_band * pos_flags
        pos_band_counts = np.bincount(pos_rows, weights=pos_in_band, minlength=n_users)
        pos_flagged_counts = np.bincount(pos_rows, weights=pos_in_band_flagged, minlength=n_users)
        user_counts[:, band_idx, -1, 0] = band_counts[home_nodes] - pos_band_counts
        user_counts[:, band_idx, -1, 1] = flagged_counts[home_nodes] - pos_flagged_counts

    count_sums = user_counts.sum(axis=0)
    boot_sums = _bootstrap_sums(user_counts.reshape(n_users, -1), n_bootstrap).reshape(
        n_bootstrap, n_bands, len(FAR_GROUPS), 2
    )
    shares = count_sums[..., 1] / count_sums[..., 0]
    boot_shares = boot_sums[..., 1] / boot_sums[..., 0]

    rows = []
    for band_idx, (band_label, _, _) in enumerate(far_bands):
        for group_idx, group in enumerate(FAR_GROUPS):
            share_samples = boot_shares[:, band_idx, group_idx]
            rows.append({
                "hop_band": band_label,
                "group": group,
                "n": count_sums[band_idx, group_idx, 0],
                "share": shares[band_idx, group_idx],
                "share_lo": np.nanquantile(share_samples, 0.025),
                "share_hi": np.nanquantile(share_samples, 0.975),
            })

    return pl.DataFrame(rows)


def with_intervals(frame: pl.DataFrame, names: list[str], precision: int = 2) -> pl.DataFrame:
    """Replace each column ``name`` of ``names`` and its ``_lo`` and ``_hi`` bounds by one string ``"x [lo, hi]"``."""
    interval_cols = []
    for name in names:
        value, low, high = (pl.col(col).round(precision).cast(pl.String) for col in (name, f"{name}_lo", f"{name}_hi"))
        interval_cols.append(pl.format("{} [{}, {}]", value, low, high).alias(name))

    bound_names = [f"{name}{suffix}" for name in names for suffix in ("_lo", "_hi")]
    return frame.with_columns(interval_cols).drop(bound_names)


def network_tables(network: Network, run: DatasetRun) -> dict[str, pl.DataFrame]:
    """Return the same-line and far-positive tables of one network, keyed by their title, with bounds as strings."""
    ratio_names = ["line_ratio", "near_ratio", "line_given_near_ratio"]
    no_zones = np.zeros_like(run.popular_zones)
    all_zones_enrichment = same_line_enrichment(network, run, no_zones)
    unpopular_enrichment = same_line_enrichment(network, run, run.popular_zones)
    matched_enrichment = direction_matched_enrichment(network, run, run.popular_zones)
    matched_names = ["line_band", "line_sector", "line_given_near_band", "line_given_near_sector"]

    all_users = np.ones(len(run.home_nodes), dtype=bool)
    served_flags = np.broadcast_to(network.served_zones, run.hops.shape)
    served_shares = far_positive_shares(served_flags, run, all_users)
    served_home_users = network.served_zones[run.home_nodes]
    line_shares = far_positive_shares(network.home_line_zones, run, served_home_users)

    return {
        "Same-line enrichment, all zones": with_intervals(all_zones_enrichment, ratio_names),
        f"Same-line enrichment, popularity top {TOP_K} zones left out": with_intervals(
            unpopular_enrichment, ratio_names
        ),
        f"Direction-matched enrichment, popularity top {TOP_K} zones left out": with_intervals(
            matched_enrichment, matched_names
        ),
        "Far positives, share near a point, all test users": with_intervals(served_shares, ["share"]),
        "Far positives, share on a home line, users with a served home": with_intervals(line_shares, ["share"]),
    }


def network_summary(network: Network, run: DatasetRun) -> dict[str, int]:
    """Return the number of network points near a zone, of zones served and of test users with a served home."""
    served_zones = network.served_zones
    point_counts = np.asarray(network.near_zone.sum(axis=0)).ravel()

    return {
        "points_near_zone": int((point_counts > 0).sum()),
        "served_zones": int(served_zones.sum()),
        "zones": len(served_zones),
        "users_served_home": int(served_zones[run.home_nodes].sum()),
        "users": len(run.home_nodes),
    }


def check_positive_hops(run: DatasetRun) -> int:
    """Return the number of positives whose saved hop distance differs from the contiguity hops of the graph cache."""
    pos_homes = run.positives["home_node"].to_numpy()
    pos_nodes = run.positives["pos_node"].to_numpy()
    saved_hops = run.positives["pos_hops"].to_numpy()

    return int((run.hops[pos_homes, pos_nodes] != saved_hops).sum())


def _contiguity_hops(loc_ids: list, edges: gpd.GeoDataFrame) -> np.ndarray:
    """Return the ``[N, N]`` contiguity-hop distance between the nodes, ordered like ``loc_ids``."""
    node_index = {loc_id: idx for idx, loc_id in enumerate(loc_ids)}
    sources = [node_index[loc_id] for loc_id in edges.index.get_level_values(0)]
    targets = [node_index[loc_id] for loc_id in edges.index.get_level_values(1)]
    n_nodes = len(loc_ids)
    adjacency = sp.csr_matrix((np.ones(len(sources)), (sources, targets)), shape=(n_nodes, n_nodes))

    return shortest_path(adjacency, method="D", unweighted=True, directed=False)


def _run_path(cfg: Config, data: str) -> Path:
    """Return the directory of the run of ``data`` in ``RUNS``."""
    return run_dir(PROJECT_ROOT / cfg.paths.runs, cfg.data.name, RUNS[data])


def _popularity_rows(run_path: Path) -> pl.DataFrame:
    """Return the per-user rows of the popularity baseline.

    The test-user column of the run is kept as ``run_user``, and ``user`` numbers the test users from 0.
    """
    per_user = pl.read_parquet(run_path / "per-user.parquet").rename({"user_idx": "user"})
    popularity_rows = per_user.filter((pl.col("name") == POPULARITY_MODEL) & (pl.col("stage") == PER_USER_STAGE))
    user_codes = popularity_rows["user"].rank("dense").cast(pl.Int64) - 1

    return popularity_rows.with_columns(run_user=pl.col("user"), user=user_codes)


def _popular_zones(run_path: Path) -> np.ndarray:
    """Return ``[N]`` bool, true for the ``TOP_K`` zones scored highest by the popularity baseline of a run.

    The popularity baseline gives every test user the same score vector, so the first one is read.
    """
    score_rows = pl.scan_parquet(run_path / "scores.parquet").filter(pl.col("name") == POPULARITY_MODEL)
    popularity_scores = score_rows.head(1).collect()["scores"][0].to_numpy()

    popular_zones = np.zeros(len(popularity_scores), dtype=bool)
    popular_zones[np.argsort(-popularity_scores)[:TOP_K]] = True
    return popular_zones


def _home_distance_matrix(user_homes: pl.DataFrame, distances: pl.DataFrame, n_nodes: int) -> np.ndarray:
    """Return the ``[N, N]`` distance from each home node of a test user to every node, NaN for the other rows."""
    home_rows = user_homes.join(distances, on="run_user").unique("home_node")

    home_distance = np.full((n_nodes, n_nodes), np.nan)
    for home_node, distance_list in zip(home_rows["home_node"], home_rows["distances"], strict=True):
        home_distance[home_node] = distance_list.to_numpy()

    return home_distance


def _build_network(name: str, points: pl.DataFrame, point_lines: pl.DataFrame, polygons: gpd.GeoSeries) -> Network:
    """Return the ``Network`` of ``points`` (``point_id, lon, lat``) on the lines of ``point_lines``."""
    point_geometry = gpd.points_from_xy(points["lon"], points["lat"], crs=4326)
    point_frame = gpd.GeoDataFrame(geometry=point_geometry).to_crs(polygons.crs)
    zone_frame = gpd.GeoDataFrame(geometry=polygons)
    near_pairs = point_frame.sjoin(zone_frame, predicate="dwithin", distance=RADIUS_M)

    n_points = len(points)
    near_values = np.ones(len(near_pairs))
    near_rows = near_pairs["index_right"].to_numpy()
    near_cols = near_pairs.index.to_numpy()
    near_zone = sp.csr_matrix((near_values, (near_rows, near_cols)), shape=(len(polygons), n_points))

    point_index = {point_id: idx for idx, point_id in enumerate(points["point_id"])}
    line_calls = point_lines.filter(pl.col("point_id").is_in(list(point_index)))
    call_points = line_calls["point_id"].replace_strict(point_index).to_numpy()
    call_lines = line_calls["line_id"].rank("dense").cast(pl.Int64).to_numpy() - 1
    n_lines = int(call_lines.max()) + 1
    point_line = sp.csr_matrix((np.ones(len(call_points)), (call_points, call_lines)), shape=(n_points, n_lines))
    same_line = ((point_line @ point_line.T) > 0).astype(np.float64)
    same_line = (same_line + sp.identity(n_points, format="csr") > 0).astype(np.float64)

    return Network(name=name, near_zone=near_zone, same_line=same_line.tocsr())


def _motorway_features(clip_path: Path) -> gpd.GeoDataFrame:
    """Return the motorway junctions and the motorway ways of an OSM extract, with their tags, in WGS 84."""
    with tempfile.TemporaryDirectory() as tmp_name:
        filtered_path = Path(tmp_name) / "motorway.osm.pbf"
        export_path = Path(tmp_name) / "motorway.geojsonseq"
        filter_command = ["osmium", "tags-filter", str(clip_path), "n/highway=motorway_junction", "w/highway=motorway"]
        export_command = ["osmium", "export", "-f", "geojsonseq", str(filtered_path)]
        subprocess.run([*filter_command, "-o", str(filtered_path)], check=True, capture_output=True)
        subprocess.run([*export_command, "-o", str(export_path)], check=True, capture_output=True)
        features = gpd.read_file(export_path, engine="pyogrio", columns=["highway", "ref"])

    return features.set_crs(4326, allow_override=True)


def _user_rows(user_mask: np.ndarray, home_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the users kept by ``user_mask`` ``[U]``, their nodes in ``home_nodes`` ``[U]``, and the row of each user.

    The row is -1 for the users left out.
    """
    user_rows = np.flatnonzero(user_mask)
    row_of_user = np.full(len(user_mask), -1)
    row_of_user[user_rows] = np.arange(len(user_rows))

    return user_rows, home_nodes[user_mask], row_of_user


def _distance_bands(run: DatasetRun) -> np.ndarray:
    """Return ``[N, N]``, the band of ``cfg.data.distance_bands`` of every node from each home node."""
    band_edges = np.asarray(run.cfg.data.distance_bands, dtype=np.float64)
    return np.digitize(run.home_distance / 1000.0, band_edges)


def _direction_sectors(centroids: np.ndarray) -> np.ndarray:
    """Return ``[N, N]``, the sector of node ``n`` seen from node ``h``.

    The sectors are ``N_SECTORS`` equal directions, numbered clockwise from north.
    """
    offsets = centroids[None, :, :] - centroids[:, None, :]
    bearings = np.arctan2(offsets[..., 0], offsets[..., 1]) % (2 * np.pi)
    sectors = np.floor(bearings / (2 * np.pi / N_SECTORS)).astype(np.int64)

    return np.minimum(sectors, N_SECTORS - 1)


def _home_stratum_counts(
    node_strata: np.ndarray, n_strata: int, home_line: np.ndarray, served_zones: np.ndarray, kept_zones: np.ndarray
) -> np.ndarray:
    """Return ``[N, n_strata, 3]``: per home, the kept nodes in each stratum, those on a home line, and those served.

    The home node itself is left out.

    Args:
        node_strata: ``[N, N]`` stratum in ``[0, n_strata)`` of node ``n`` for home ``h``.
        n_strata: Number of strata.
        home_line: ``[N, N]`` bool, node ``n`` on a line through home ``h``.
        served_zones: ``[N]`` bool, zones near a point.
        kept_zones: ``[N]`` bool, zones counted.
    """
    n_nodes = node_strata.shape[0]
    counted_nodes = ~np.eye(n_nodes, dtype=bool) & kept_zones[None, :]
    flat_strata = (np.arange(n_nodes)[:, None] * n_strata + node_strata).ravel()

    stratum_counts = []
    for node_flags in (counted_nodes, counted_nodes & home_line, counted_nodes & served_zones[None, :]):
        flag_weights = node_flags.ravel().astype(np.float64)
        stratum_counts.append(np.bincount(flat_strata, weights=flag_weights, minlength=n_nodes * n_strata))

    return np.stack(stratum_counts, axis=-1).reshape(n_nodes, n_strata, 3)


def _matched_counts(pos_counts: np.ndarray, unv_counts: np.ndarray) -> np.ndarray:
    """Return the ``MATCHED_COUNTS`` of each user and stratum from the positive and unvisited counts ``[U, K, 3]``.

    The last axis of both inputs holds the nodes, those on a home line and those near a point.
    """
    pos_n, pos_line, pos_near = pos_counts[..., 0], pos_counts[..., 1], pos_counts[..., 2]
    unv_n, unv_line, unv_near = unv_counts[..., 0], unv_counts[..., 1], unv_counts[..., 2]
    has_unv = unv_n > 0
    has_unv_near = unv_near > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        line_rate = np.where(has_unv, unv_line / unv_n, 0.0)
        line_near_rate = np.where(has_unv_near, unv_line / unv_near, 0.0)

    return np.stack(
        [
            np.where(has_unv, pos_n, 0.0),
            np.where(has_unv, pos_line, 0.0),
            pos_n * line_rate,
            np.where(has_unv_near, pos_line, 0.0),
            pos_near * line_near_rate,
        ],
        axis=-1,
    )


def _bootstrap_sums(user_counts: np.ndarray, n_bootstrap: int, seed: int = 0, chunk: int = 200) -> np.ndarray:
    """Return ``[n_bootstrap, K]`` column sums of ``user_counts`` ``[U, K]`` over users resampled with replacement."""
    rng = np.random.default_rng(seed)
    n_users = user_counts.shape[0]
    user_probs = np.full(n_users, 1.0 / n_users)

    sum_chunks = []
    for start in range(0, n_bootstrap, chunk):
        chunk_size = min(chunk, n_bootstrap - start)
        resample_weights = rng.multinomial(n_users, user_probs, size=chunk_size).astype(np.float64)
        sum_chunks.append(resample_weights @ user_counts)

    return np.concatenate(sum_chunks)


def _share_ratio(sums: np.ndarray, pos_num: int, pos_den: int, unv_num: int, unv_den: int) -> np.ndarray:
    """Return the positive share over the unvisited share, from summed counts indexed on their last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_share = sums[..., pos_num] / sums[..., pos_den]
        unv_share = sums[..., unv_num] / sums[..., unv_den]
        return pos_share / unv_share


def _band_labels(bands: list[float]) -> list[str]:
    """Return the label of every distance band in km, e.g. ``"1-2.5"``, the last one ``"10+"``."""
    edge_labels = [f"{edge:g}" for edge in [0.0, *bands]]
    inner_labels = [f"{low}-{high}" for low, high in zip(edge_labels[:-1], edge_labels[1:], strict=True)]

    return [*inner_labels, f"{edge_labels[-1]}+"]


def main() -> None:
    """Print every table for every dataset."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(40)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    for data in RUNS:
        run = read_dataset_run(data)
        print(f"\n=== {data}: hop mismatches {check_positive_hops(run)}")
        for network in (rail_network(run.cfg, run.polygons), motorway_network(data, run.polygons)):
            print(f"\n--- {data} {network.name}: {network_summary(network, run)}")
            for title, table in network_tables(network, run).items():
                print(title)
                print(table)


if __name__ == "__main__":
    main()
