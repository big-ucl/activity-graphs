"""Overture Maps data loader for POI counts and land-use features."""

from pathlib import Path
from typing import Self

import city2graph as c2g
import geopandas as gpd
import pandas as pd

from activitygraphs.base import CRS
from activitygraphs.config import OverturePaths


class Overture:
    """Overture Maps feature extractor; use ``Overture.load()`` to download and cache data."""

    def __init__(self, land_use: gpd.GeoDataFrame, place: gpd.GeoDataFrame):
        self.land_use = land_use
        self.place = place

    def add_poi_counts(
        self,
        locations: gpd.GeoDataFrame,
        normalise: bool = True,
        category: str = "top_category",
    ) -> gpd.GeoDataFrame:
        """Add Overture place POIs to ``locations``, in one column per category.

        Args:
            locations: Polygon locations to enrich; must have a ``loc_id`` column and polygon geometry.
            normalise: If True, divide counts by polygon area (POIs per m^2).
            category: Taxonomy field to use as the pivot column (default ``"top_category"``).

        Returns:
            ``locations`` with additional ``poi_<category>`` columns.
        """
        utm_crs = locations.estimate_utm_crs()

        original_locations = locations
        locations = locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
        locations["area"] = locations.geometry.area

        places = self.place.copy()
        places[category] = (
            places["taxonomy"].astype(str).str.extract(r'hierarchy[\'"]:\s?\[[\'"]([^\'"]+)[\'"]')
        )  # Extract the first element in the "hierarchy"

        intersection = locations.sjoin(places.to_crs(utm_crs), predicate="intersects", how="left")
        poi_by_sector = pd.DataFrame(intersection.groupby(["loc_id", category]).size(), columns=["count"]).reset_index()

        if normalise:
            poi_by_sector = locations[["loc_id", "area"]].merge(poi_by_sector, on="loc_id", how="left")
            poi_by_sector["count"] = poi_by_sector["count"] / poi_by_sector["area"]

        poi_counts = pd.pivot_table(
            poi_by_sector,
            columns=category,
            index="loc_id",
            values="count",
            fill_value=0,
        ).add_prefix("poi_")
        enriched_locations = original_locations.merge(poi_counts, on="loc_id", how="left").fillna(0)

        return enriched_locations

    def add_land_uses(
        self,
        locations: gpd.GeoDataFrame,
        normalise: bool = True,
    ) -> gpd.GeoDataFrame:
        """Add ``land_use_<subtype>`` columns representing location area taken by land-use subtype.

        Args:
            locations: Must have Polygon or MultiPolygon geometry.
            normalise: If True, express each land-use column as fraction of total area.
        """
        if not locations.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError("Locations geometries must all be Polygons or MultiPolygons")

        utm_crs = locations.estimate_utm_crs()
        original_locations = locations

        locations = locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
        locations["total_area"] = locations.geometry.area

        intersection = locations.overlay(self.land_use.to_crs(utm_crs), how="intersection")
        intersection["land_use_area"] = intersection.geometry.area
        land_use_by_sector = (
            intersection
            .drop(columns=["geometry"])
            .groupby(["loc_id", "subtype"])
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={"subtype": "land_use"}, errors="raise")
        )

        if normalise:
            land_use_by_sector["pct_area"] = land_use_by_sector["land_use_area"] / land_use_by_sector["total_area"]

        value_col = "pct_area" if normalise else "land_use_area"
        loc_land_uses = pd.pivot_table(
            land_use_by_sector,
            columns="land_use",
            index="loc_id",
            values=value_col,
            fill_value=0,
        ).add_prefix("land_use_")
        loc_land_uses = original_locations.merge(loc_land_uses, on="loc_id", how="left").fillna(0)

        return loc_land_uses

    @classmethod
    def load(
        cls, locations: gpd.GeoDataFrame | gpd.GeoSeries, cfg: OverturePaths, project_root: Path | None = None
    ) -> Self:
        """Load Overture land-use and place data from disk, downloading via city2graph if not cached."""
        project_root = project_root if project_root is not None else Path(".")

        output_dir = project_root / cfg.directory
        land_use_path = output_dir / cfg.land_use
        places_path = output_dir / cfg.place

        geometry = locations["geometry"] if isinstance(locations, gpd.GeoDataFrame) else locations

        if not (land_use_path.exists() and places_path.exists()):
            output_dir.mkdir(parents=True, exist_ok=True)

            # noinspection PyTypeChecker
            c2g.load_overture_data(
                area=geometry.to_crs(CRS).union_all(),  # pyrefly: ignore[bad-argument-type]
                types=["land_use", "place"],
                output_dir=str(output_dir),
                save_to_file=True,
            )

        land_use = gpd.read_file(land_use_path)[["id", "subtype", "class", "geometry"]].set_index("id")
        land_use = land_use[land_use.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        land_use = land_use.explode(index_parts=False)

        places = gpd.read_file(places_path)[["id", "names", "basic_category", "taxonomy", "geometry"]].set_index("id")

        return cls(land_use, places)
