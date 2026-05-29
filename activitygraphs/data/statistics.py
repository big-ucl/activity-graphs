"""Census population and employment statistics enrichment for network nodes."""

from pathlib import Path

import geopandas as gpd
import polars as pl

from activitygraphs.config import GenevaStatsInputs, TorontoStatsInputs
from activitygraphs.utils import gdf_to_polars


def add_geneva_population_job_statistics(
    locations: gpd.GeoDataFrame, cfg: GenevaStatsInputs, normalise: bool = True, project_root: Path | None = None
) -> gpd.GeoDataFrame:
    """Add population and job counts to Geneva subsector locations

    Args:
        locations: Subsector locations (must have a ``loc_id`` column).
        cfg: Geneva census file config.
        normalise: If True, divide counts by polygon area to obtain densities.
        project_root: Repo root; defaults to ``Path(".")``.
    """
    project_root: Path = project_root if project_root is not None else Path(".")

    utm_crs = locations.estimate_utm_crs()
    original_locations = locations

    statistics = gpd.read_file(project_root / cfg.directory / cfg.file)
    statistics["population"] = statistics["D_POP_HA"] * statistics["GEOM_AREA"] / 10000
    statistics["jobs"] = statistics["D_EMP_HA"] * statistics["GEOM_AREA"] / 10000
    statistics = statistics.rename(columns={"GRID_ID": "id"}).to_crs(utm_crs)
    statistics = statistics[["id", "population", "jobs", "geometry"]]

    locations = locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
    locations["area"] = locations.geometry.area

    intersection = locations.sjoin(statistics.to_crs(utm_crs), how="left", predicate="intersects")
    stats_by_sector = intersection.groupby("loc_id")[["population", "jobs"]].sum()

    stats_by_sector = original_locations.merge(stats_by_sector, on="loc_id", how="right")
    stats_by_sector["area"] = stats_by_sector.to_crs(utm_crs).geometry.area

    return normalise_statistics(stats_by_sector, normalise)


def add_toronto_population_job_statistics(
    locations: gpd.GeoDataFrame, cfg: TorontoStatsInputs, normalise: bool = True, project_root: Path | None = None
) -> gpd.GeoDataFrame:
    """Add census population and job counts to Toronto (CT/subsector) locations.

    Args:
        locations: toronto locations (must have a ``loc_id`` matching ``ALT_GEO_CODE``).
        cfg: Toronto census file config.
        normalise: If True, divide counts by area to obtain densities.
        project_root: Repo root; defaults to ``Path(".")``.
    """
    project_root: Path = project_root if project_root is not None else Path(".")

    locations_df = gdf_to_polars(locations).lazy()

    # Load census data to find population and area in km^2
    census_stats = pl.scan_csv(
        project_root / cfg.directory / cfg.population,
        encoding="utf8-lossy",
        schema_overrides={"ALT_GEO_CODE": str},
    )

    pop_stats = census_stats.filter(CHARACTERISTIC_ID=1).select("ALT_GEO_CODE", population="C1_COUNT_TOTAL")
    area_stats = census_stats.filter(CHARACTERISTIC_ID=7).select("ALT_GEO_CODE", area=pl.col("C1_COUNT_TOTAL") * 10**6)

    pop_stats = (
        locations_df.select("loc_id").join(pop_stats, left_on="loc_id", right_on="ALT_GEO_CODE").collect().to_pandas()
    )
    area_stats = (
        locations_df.select("loc_id").join(area_stats, left_on="loc_id", right_on="ALT_GEO_CODE").collect().to_pandas()
    )

    # Load census jobs data to find places of work
    dguids_to_loc_ids = census_stats.select("DGUID", loc_id="ALT_GEO_CODE")

    job_stats = pl.scan_csv(
        project_root / cfg.directory / cfg.jobs,
        schema_overrides={"ALT_GEO_CODE": str},
    ).join(dguids_to_loc_ids, on="DGUID")

    job_stats = (
        job_stats
        .filter(
            pl.col("Industry - Sectors - North American Industry Classification System (NAICS) 2017 (21)")
            == "Total - Industry - Sectors - North American Industry Classification System (NAICS) 2017",
            pl.col("Age (15A)") == "Total - Age",
            pl.col("Gender (3)") == "Total - Gender",
            pl.col("Place of work status (7)") == "Total - Place of work status",
        )
        .unique("DGUID")
        .select(
            "loc_id",
            jobs="Work activity during the reference year (4A):Total - Work activity during the reference year[1]",
        )
    )

    job_stats = locations_df.select("loc_id").join(job_stats, on="loc_id").collect().to_pandas()

    stats_by_sector = (
        locations
        .merge(pop_stats, on="loc_id", how="left")
        .merge(job_stats, on="loc_id", how="left")
        .merge(area_stats, on="loc_id", how="left")
    )

    return normalise_statistics(stats_by_sector, normalise)


def normalise_statistics(stats_by_sector: gpd.GeoDataFrame, normalise: bool = True) -> gpd.GeoDataFrame:
    """Divide ``population`` and ``jobs`` columns by ``area`` if ``normalise`` is True."""
    if normalise:
        stats_by_sector = stats_by_sector.copy()
        stats_by_sector["population"] = stats_by_sector["population"] / stats_by_sector["area"]
        stats_by_sector["jobs"] = stats_by_sector["jobs"] / stats_by_sector["area"]

    return stats_by_sector
