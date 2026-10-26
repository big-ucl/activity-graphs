"""Helpers shared by the census population and employment enrichment of each survey's zones."""

import geopandas as gpd


def normalise_statistics(stats_by_sector: gpd.GeoDataFrame, normalise: bool = True) -> gpd.GeoDataFrame:
    """Divide ``population`` and ``jobs`` columns by ``area`` if ``normalise`` is True."""
    if normalise:
        stats_by_sector = stats_by_sector.copy()
        stats_by_sector["population"] = (stats_by_sector["population"] / stats_by_sector["area"]).fillna(0)
        stats_by_sector["jobs"] = (stats_by_sector["jobs"] / stats_by_sector["area"]).fillna(0)

    return stats_by_sector
