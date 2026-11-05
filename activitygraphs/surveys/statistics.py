"""Helpers shared by the census population and employment enrichment of each survey's zones."""

from collections.abc import Sequence

import geopandas as gpd
import pandas as pd


def normalise_statistics(
    stats_by_sector: gpd.GeoDataFrame, normalise: bool = True, columns: Sequence[str] = ("population", "jobs")
) -> gpd.GeoDataFrame:
    """Divide ``columns``, ``population`` and ``jobs`` by default, by ``area`` if ``normalise`` is True."""
    if normalise:
        stats_by_sector = stats_by_sector.copy()
        for column in columns:
            stats_by_sector[column] = (stats_by_sector[column] / stats_by_sector["area"]).fillna(0)

    return stats_by_sector


def overlap_counts(zones: gpd.GeoDataFrame, units: gpd.GeoDataFrame, column: str) -> pd.Series:
    """Return the ``column`` count of ``units`` in each zone, each unit's count split by its share of area in the zone.

    Args:
        zones: Zones with a ``loc_id`` column and polygon geometry, in a projected CRS.
        units: Census units with a ``column`` of counts and polygon geometry.
        column: The count column of ``units``.

    Returns:
        The count of each zone, indexed by ``loc_id`` in the order of ``zones``. A zone that overlaps no unit counts 0.
    """
    units = units[[column, "geometry"]].to_crs(zones.crs)  # pyrefly: ignore[bad-argument-type]
    units["geometry"] = units.geometry.make_valid(method="structure", keep_collapsed=False)
    units["unit_area"] = units.area

    pieces = gpd.overlay(units, zones[["loc_id", "geometry"]], how="intersection", keep_geom_type=True)
    pieces["count"] = pieces[column] * pieces.area / pieces["unit_area"]

    return pieces.groupby("loc_id")["count"].sum().reindex(zones["loc_id"], fill_value=0.0)
