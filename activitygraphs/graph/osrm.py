"""Car, walk and bike travel times between every pair of zones, routed by OSRM in Docker over the OSM extract."""

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from activitygraphs.base import CRS
from activitygraphs.graph.rail import EDGE_INDEX_NAMES
from activitygraphs.graph.travel_times import MAX_TRAVEL_TIME, WALK_SPEED_KMH, walkable_candidate_points

log = logging.getLogger(__name__)

OSRM_IMAGE = "ghcr.io/project-osrm/osrm-backend:v6.0.0"
"""Docker image of OSRM, with the routing profiles in ``/opt``."""

OSRM_PROFILES = {"car": "car.lua", "walk": "foot.lua", "bike": "bicycle.lua"}
"""File name in ``/opt`` of the OSRM profile of each mode."""

FOOT_SPEED_LINE = "local walking_speed = 5"
"""Line of the OSRM foot profile that sets the walking speed in km/h."""

NETWORK_NAME = "network"
"""Base name of the OSRM files of a mode, ``network.osrm.*``."""

TABLE_BLOCK_SIZE = 100
"""Number of origins per OSRM table request. Every request holds all zones as destinations."""

MAX_RELINK_CANDIDATES = 300
"""Largest number of walkable OSM ways tried, nearest first, for a zone that reaches no other zone."""

NUM_TEST_ZONES_FOR_RELINK = 50
"""Number of zones routed to from each point tried for a zone that reaches no other zone. The zone is moved to the first
point tried that reaches one of these zones."""

SERVER_START_TIMEOUT_S = 300
"""Longest wait in seconds for an OSRM server to answer after it starts."""


def build_road_travel_times(nodes_gdf: gpd.GeoDataFrame, osm_path: Path, osrm_dir: Path) -> pd.DataFrame:
    """Return the car, walk and bike travel time between every pair of zones, routed by OSRM.

    Car times are free-flow. The walk profile walks at ``WALK_SPEED_KMH``, the speed of the walks routed by R5. The
    bike profile keeps the OSRM defaults. The OSRM files of each mode are built once and cached in ``osrm_dir``. A
    zone that reaches no other zone by a mode is moved, for that mode, to the nearest walkable OSM way that can reach
    another zone, and routed again.

    Args:
        nodes_gdf: Zones indexed by ``loc_id``, with the zone point in ``geometry``.
        osm_path: The OSM extract of the street network.
        osrm_dir: Directory of the OSRM files, one subdirectory per mode.

    Returns:
        One row per ordered pair of zones, indexed by the (``source``, ``target``) ``loc_id`` of the origin and the
        destination, with ``{mode}_time_min`` and ``{mode}_reachable`` for each mode of ``OSRM_PROFILES``. A pair not
        reached within ``MAX_TRAVEL_TIME`` takes ``MAX_TRAVEL_TIME`` and is flagged unreachable.
    """
    zone_points = nodes_gdf.geometry.to_crs(CRS)
    zone_ids = nodes_gdf.index.to_numpy()
    zone_coords = _coord_strings(zone_points)

    all_rows = np.arange(len(zone_ids))

    mode_times = []
    for mode in OSRM_PROFILES:
        mode_dir = osrm_dir / mode
        _prepare_network(mode, osm_path, mode_dir)

        with _osrm_server(mode_dir, len(zone_ids)) as base_url:
            durations_s = _route_table(base_url, zone_coords, all_rows, None)

            # Handle unlinked zones (zones that cannot reach other zones)
            unlinked_rows = _unlinked_rows(durations_s)
            if unlinked_rows.size > 0:
                unlinked_points = gpd.GeoSeries(zone_points.iloc[unlinked_rows], crs=CRS)
                candidates_by_zone = walkable_candidate_points(unlinked_points, osm_path, MAX_RELINK_CANDIDATES)

                linked_rows = np.setdiff1d(all_rows, unlinked_rows)[:NUM_TEST_ZONES_FOR_RELINK]
                test_coords = [zone_coords[row] for row in linked_rows]

                moved_coords = list(zone_coords)
                moved_distances = {}
                for row in unlinked_rows:
                    zone_id = zone_ids[row]
                    candidate_points, candidate_distance_m = candidates_by_zone[zone_id]
                    candidate_coords = _coord_strings(candidate_points)
                    linked_idx = _first_linked_candidate(base_url, candidate_coords, test_coords)
                    if linked_idx is not None:
                        moved_coords[row] = candidate_coords[linked_idx]
                        moved_distances[zone_id] = round(float(candidate_distance_m[linked_idx]))

                log.info(
                    "Zone points moved for %s to the nearest walkable OSM way that reaches another zone, in metres: %s",
                    mode,
                    moved_distances,
                )
                durations_s[unlinked_rows, :] = _route_table(base_url, moved_coords, unlinked_rows, None)
                durations_s[:, unlinked_rows] = _route_table(base_url, moved_coords, all_rows, unlinked_rows)

        still_unlinked_ids = zone_ids[_unlinked_rows(durations_s)].tolist()
        if still_unlinked_ids:
            log.warning("Some zones cannot reach any other zone by %s: %s", mode, still_unlinked_ids)

        mode_times.append(_duration_table(durations_s, zone_ids, mode))

    return pd.concat(mode_times, axis=1)


def _prepare_network(mode: str, osm_path: Path, mode_dir: Path) -> None:
    """Build the OSRM files of ``mode`` in ``mode_dir``, unless they are already there."""
    network_base = mode_dir / f"{NETWORK_NAME}.osrm"
    if Path(f"{network_base}.mldgr").exists():
        return

    mode_dir.mkdir(parents=True, exist_ok=True)
    linked_extract = mode_dir / f"{NETWORK_NAME}.osm.pbf"
    linked_extract.unlink(missing_ok=True)
    try:
        os.link(osm_path, linked_extract)
    except OSError:
        shutil.copyfile(osm_path, linked_extract)

    profile_mount = []
    if mode == "walk":
        foot_profile = mode_dir / "foot.lua"
        foot_profile.write_text(_foot_profile(_read_image_file("/opt/foot.lua")))
        profile_mount = ["-v", f"{foot_profile}:/opt/foot.lua:ro"]

    profile_path = f"/opt/{OSRM_PROFILES[mode]}"
    container_base = f"/data/{NETWORK_NAME}.osrm"
    log.info("Building the OSRM network of %s in %s", mode, mode_dir)
    _run_osrm(mode_dir, profile_mount, ["osrm-extract", "-p", profile_path, f"/data/{NETWORK_NAME}.osm.pbf"])
    _run_osrm(mode_dir, [], ["osrm-partition", container_base])
    _run_osrm(mode_dir, [], ["osrm-customize", container_base])
    linked_extract.unlink()


def _foot_profile(profile_text: str) -> str:
    """Return the OSRM foot profile with its walking speed set to ``WALK_SPEED_KMH``.

    Raises:
        ValueError: The profile does not hold ``FOOT_SPEED_LINE`` exactly once.
    """
    if profile_text.count(FOOT_SPEED_LINE) != 1:
        raise ValueError(f"The OSRM foot profile does not set its walking speed with `{FOOT_SPEED_LINE}`.")

    return profile_text.replace(FOOT_SPEED_LINE, f"local walking_speed = {WALK_SPEED_KMH}")


def _read_image_file(path: str) -> str:
    """Return the text of a file of ``OSRM_IMAGE``."""
    result = subprocess.run(["docker", "run", "--rm", OSRM_IMAGE, "cat", path], check=True, capture_output=True)
    return result.stdout.decode()


def _run_osrm(mode_dir: Path, extra_mounts: list[str], command: list[str]) -> None:
    """Run an OSRM command in ``OSRM_IMAGE`` as the current user, with ``mode_dir`` mounted at ``/data``."""
    user = f"{os.getuid()}:{os.getgid()}"
    docker_command = ["docker", "run", "--rm", "--user", user, "-v", f"{mode_dir}:/data", *extra_mounts, OSRM_IMAGE]
    subprocess.run([*docker_command, *command], check=True, capture_output=True)


@contextmanager
def _osrm_server(mode_dir: Path, num_zones: int) -> Generator[str]:
    """Run an OSRM server on the network in ``mode_dir`` and yield its base URL. The server stops on exit."""
    port = _free_port()
    container_name = f"osrm-{mode_dir.parent.parent.name.lower()}-{mode_dir.name}-{port}"
    user = f"{os.getuid()}:{os.getgid()}"
    max_table_size = str(num_zones + 1)
    server_command = [
        "docker", "run", "-d", "--rm", "--name", container_name, "--user", user,
        "-p", f"127.0.0.1:{port}:5000", "-v", f"{mode_dir}:/data", OSRM_IMAGE,
        "osrm-routed", "--algorithm", "mld", "--max-table-size", max_table_size, f"/data/{NETWORK_NAME}.osrm",
    ]  # fmt: skip
    subprocess.run(server_command, check=True, capture_output=True)

    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url)
        yield base_url
    finally:
        subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)


def _coord_strings(points: gpd.GeoSeries) -> list[str]:
    """Return each of ``points`` as the ``lon,lat`` string of an OSRM request."""
    point_lon = points.x.to_numpy()
    point_lat = points.y.to_numpy()

    return [f"{lon:.6f},{lat:.6f}" for lon, lat in zip(point_lon, point_lat, strict=True)]


def _route_table(base_url: str, coords: list[str], sources: np.ndarray, destinations: np.ndarray | None) -> np.ndarray:
    """Return the OSRM duration in seconds from each of ``sources`` to each destination, ``NaN`` where no route is found.

    Args:
        base_url: The base URL of the OSRM server.
        coords: The ``lon,lat`` string of every zone.
        sources: The positions in ``coords`` of the origins.
        destinations: The positions in ``coords`` of the destinations. ``None`` takes every zone.

    Returns:
        Durations, ``[num_sources, num_destinations]``.
    """
    all_coords = ";".join(coords)
    destination_param = "" if destinations is None else "&destinations=" + ";".join(map(str, destinations))

    blocks = []
    for block_start in range(0, len(sources), TABLE_BLOCK_SIZE):
        block_sources = ";".join(map(str, sources[block_start : block_start + TABLE_BLOCK_SIZE]))
        url = (
            f"{base_url}/table/v1/driving/{all_coords}?sources={block_sources}{destination_param}&annotations=duration"
        )
        with urllib.request.urlopen(url, timeout=600) as response:
            table = json.load(response)
        block_durations = [[np.nan if value is None else value for value in row] for row in table["durations"]]
        blocks.append(np.asarray(block_durations, dtype=np.float64))

    return np.vstack(blocks)


def _first_linked_candidate(base_url: str, candidate_coords: list[str], test_coords: list[str]) -> int | None:
    """Return the position of the first point of ``candidate_coords`` from which OSRM reaches a zone of ``test_coords``.

    Return ``None`` when no point of ``candidate_coords`` reaches one of these zones.

    Args:
        base_url: The base URL of the OSRM server.
        candidate_coords: The ``lon,lat`` string of each point tried, in the order they are tried.
        test_coords: The ``lon,lat`` string of each zone routed to.
    """
    coords = [*candidate_coords, *test_coords]
    candidate_rows = np.arange(len(candidate_coords))
    test_rows = np.arange(len(candidate_coords), len(coords))
    durations_s = _route_table(base_url, coords, candidate_rows, test_rows)
    reaches_any = ~np.isnan(durations_s).all(axis=1)

    if not reaches_any.any():
        return None

    return int(np.argmax(reaches_any))


def _unlinked_rows(durations_s: np.ndarray) -> np.ndarray:
    """Return the rows of a square duration matrix whose zone reaches no other zone."""
    other_durations = durations_s.copy()
    np.fill_diagonal(other_durations, np.nan)

    return np.flatnonzero(np.isnan(other_durations).all(axis=1))


def _free_port() -> int:
    """Return a TCP port free on the local host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_server(base_url: str) -> None:
    """Wait until the OSRM server at ``base_url`` answers a request, with any HTTP status.

    Raises:
        TimeoutError: The server does not answer within ``SERVER_START_TIMEOUT_S``.
    """
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/nearest/v1/driving/0,0", timeout=10):
                return
        except urllib.error.HTTPError:
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)

    raise TimeoutError(f"The OSRM server at {base_url} did not answer within {SERVER_START_TIMEOUT_S} s.")


def _duration_table(durations_s: np.ndarray, zone_ids: np.ndarray, mode: str) -> pd.DataFrame:
    """Return the OSRM durations as minutes per ordered pair of zones, capped at ``MAX_TRAVEL_TIME``.

    Args:
        durations_s: Durations in seconds, ``[num_zones, num_zones]``, ``NaN`` where no route is found.
        zone_ids: The ``loc_id`` of each zone, in the order of the rows and columns of ``durations_s``.
        mode: The mode, which names the columns.
    """
    max_time_min = MAX_TRAVEL_TIME.total_seconds() / 60
    time_min = durations_s.reshape(-1) / 60
    reachable = ~np.isnan(time_min) & (time_min <= max_time_min)
    pair_index = pd.MultiIndex.from_product([zone_ids, zone_ids], names=EDGE_INDEX_NAMES)
    mode_columns = {f"{mode}_time_min": np.where(reachable, time_min, max_time_min), f"{mode}_reachable": reachable}

    return pd.DataFrame(mode_columns, index=pair_index)
