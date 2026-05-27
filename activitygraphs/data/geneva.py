"""Geneva TPG survey loader and NetworkData subclass."""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from polars import selectors as cs
from rapidfuzz import fuzz, process

from activitygraphs import utils
from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA, Mode, Purpose
from activitygraphs.config import DataConfig, GenevaDataConfig
from activitygraphs.data.gtfs import GTFSInputs
from activitygraphs.network import (
    NetworkData,
    build_special_locations,
)
from activitygraphs.utils import DataFrameStore, add_lon_lat_from_centroid, check_schema

GENEVA_CANTON_CODE = 25

LOCATION_REGEXES = {
    "subsector": r"([\s\S]+) - sous_secteur\s*$",
    "municipality_swiss": r"([\s\S]+) - (\d\d\d\d)\s*$",
    "municipality_french": r"([\s\S]+) - (\d\d\d\d\d)\s*$",
    "na": r"NA",
}

STOP_NAME_MAPPING = {
    "Domicile": "NA",
    "Home": "NA",
    "Other": "NA",
    "Cuvat, Les Voisins": "CRUSEILLES - 74350",
    "Genève-Cornavin": "Genève",
    "Lucerne": "Luzern",
    "Saint-Julien, SNCF": "Saint-Julien-en-Genevois, SNCF",
    "Viège": "Visp",
    "Collex-Bossy - 1239": "Collex - 1239",
    "Perly-Certoux - 1258": "Perly - 1258",
    "LÉAZ - 1200": "Léaz, Village",
    "Port Noir/Genève-Plage, lac": "Genève-Port Noir (lac)",
    "Pregny-Chambésy - 1292": "Chambésy - 1292",
    "Pâquis, lac": "Genève-Pâquis (lac)",
    "Lancy - 1212": "Grand-Lancy - 1212",
    "Pringy": "Pringy - 1663",
    "Cluses": "CLUSES - 74300",
    "Signy-Avenex - 1274": "Signy, Le Glassey",
    "De-Chateaubriand, lac": "Genève-De-Châteaubriand (lac)",
    "Eaux-Vives, lac": "Genève-Eaux-Vives (lac)",
    "Saint-Cergue - 1265": "La Cure",
    "Saint-Cergue - 1264": "St-Cergue",
    "Vernier, Etang-Place": "Vernier, Etang Place",
    "Vernier, CHôtelaine": "Vernier, Châtelaine",
}

MODE_MAPPING = {
    "mode_autre": Mode.OTHER,
    "mode_bateau_navette": Mode.BOAT,
    "mode_bus": Mode.BUS,
    "mode_car_interurbain": Mode.COACH,
    "mode_marche_à_pied": Mode.WALK,
    "mode_moto_scooter": Mode.MOTORCYCLE,
    "mode_taxi_vtc": Mode.TAXI,
    "mode_train": Mode.TRAIN,
    "mode_tramway": Mode.TRAMWAY,
    "mode_trottinette": Mode.CYCLE,
    "mode_velo": Mode.CYCLE,
    "mode_voiture_conducteur": Mode.CAR,
    "mode_voiture_passager": Mode.VEH_PASS,
}

PURPOSE_MAPPING = {
    "od_lieu_achat": Purpose.SHOP,
    "od_lieu_autre": Purpose.OTHER,
    "od_lieu_autrelieutravail": Purpose.WORK_OTHER,
    "od_lieu_autreloisir": Purpose.LEISURE_OTHER,
    "od_lieu_domicile": Purpose.HOME,
    "od_lieu_etude": Purpose.STUDY,
    "od_lieu_resto": Purpose.ENTERTAINMENT,
    "od_lieu_travail": Purpose.WORK_MAIN,
    "od_lieu_visite": Purpose.VISIT,
    "od_lieu_voyage_longue_distance": Purpose.LONG_DISTANCE_TRIP,
}

FUZZY_MATCH_THRESHOLD = 65


@dataclass(frozen=True)
class GenevaInputs:
    """Parsed raw inputs for the Geneva TPG survey (journeys, boundary GeoDataFrames, and GTFS)."""

    raw_journeys_df: pl.DataFrame

    subsectors_gdf: gpd.GeoDataFrame
    postcodes_gdf: gpd.GeoDataFrame
    localities_gdf: gpd.GeoDataFrame
    swiss_boundaries_gdf: gpd.GeoDataFrame
    french_gdf: gpd.GeoDataFrame

    gtfs: GTFSInputs


class GenevaData(NetworkData, DataFrameStore):
    """Geneva TPG ``NetworkData`` subclass with caching via ``DataFrameStore``."""

    def __init__(
        self,
        inputs: GenevaInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        super().__init__(
            user_journeys_df,
            locations_gdf,
            filters,
        )

        self.inputs = inputs
        self.gtfs = self.inputs.gtfs

    def _copy(self, filters: list[str] | None = None):
        return GenevaData(self.inputs, self._locations_gdf, self._user_journeys_df, filters)

    @classmethod
    def load(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> "GenevaData":
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        gva_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)

            return cls(gva_inputs, locations_gdf, user_journeys_df)
        else:
            data = build_geneva_data(gva_inputs)
            data.save(cfg, project_root, name)

            return build_geneva_data(gva_inputs)

    def save(self, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None):
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self.locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self.user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")


def load_files(cfg: GenevaDataConfig, project_root: Path | None = None) -> GenevaInputs:
    """Read all raw Geneva files (journeys, boundaries, GTFS) into a ``GenevaInputs`` container."""

    def parse_gtfs_date(*cols: str) -> pl.Expr:
        return pl.col(*cols).cast(pl.String).str.to_date("%Y%m%d")

    project_root: Path = project_root if project_root is not None else Path(".")

    raw_path = project_root / cfg.paths.raw

    boundaries_path = project_root / cfg.inputs.boundaries.directory
    boundaries_files = cfg.inputs.boundaries

    gtfs_path = project_root / cfg.paths.gtfs
    gtfs_files = cfg.inputs.gtfs

    raw_journeys_df = pl.read_parquet(raw_path / cfg.inputs.raw_journeys)

    subsectors_gdf = gpd.read_file(boundaries_path / boundaries_files.geneva_subsectors).to_crs(CRS)
    postcodes_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_postcodes).to_crs(CRS)
    localities_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_localities).to_crs(CRS)
    boundaries_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_boundaries).to_crs(CRS)
    french_gdf = gpd.read_file(boundaries_path / boundaries_files.french_postcodes).to_crs(CRS)

    stops_df = pl.read_csv(
        gtfs_path / gtfs_files.stops,
        schema={
            "stop_id": pl.String,
            "stop_name": pl.String,
            "stop_lat": pl.Float32,
            "stop_lon": pl.Float32,
            "location_type": pl.Categorical(),
            "parent_station": pl.String,
        },
    ).with_columns(loc_id=pl.when(pl.col("parent_station") == "").then("stop_id").otherwise("parent_station"))

    stop_id_to_loc_id = stops_df.select("stop_id", "loc_id").lazy()
    stop_times_df = pl.scan_csv(gtfs_path / gtfs_files.stop_times).join(stop_id_to_loc_id, on="stop_id", how="left")
    trips_df = pl.scan_csv(gtfs_path / gtfs_files.trips)
    routes_df = pl.read_csv(gtfs_path / gtfs_files.routes)
    agency_df = pl.read_csv(gtfs_path / gtfs_files.agency)
    calendar_df = pl.read_csv(gtfs_path / gtfs_files.calendar).with_columns(parse_gtfs_date("start_date", "end_date"))
    calendar_dates_df = pl.read_csv(gtfs_path / gtfs_files.calendar_dates).with_columns(parse_gtfs_date("date"))

    transfers_df = pl.read_csv(
        gtfs_path / gtfs_files.transfers,
        schema={
            "from_stop_id": pl.String,
            "to_stop_id": pl.String,
            "transfer_type": pl.Categorical(),
            "min_transfer_time": pl.Int64,
            # All other fields are empty in the 2022 Swiss GTFS timetable, and `transfer_type` is always 2
        },
    )

    gtfs = GTFSInputs(
        stops_df, stop_times_df, trips_df, routes_df, agency_df, calendar_df, calendar_dates_df, transfers_df
    )

    return GenevaInputs(
        raw_journeys_df, subsectors_gdf, postcodes_gdf, localities_gdf, boundaries_gdf, french_gdf, gtfs
    )


def build_geneva_data(inputs: GenevaInputs) -> GenevaData:
    """Parse raw Geneva inputs into a standardised ``GenevaData`` instance."""
    # Create locations with all PT stops
    locations_gdf = build_geneva_locations(
        inputs.gtfs.stops_df,
        inputs.subsectors_gdf,
        inputs.postcodes_gdf,
        inputs.localities_gdf,
        inputs.swiss_boundaries_gdf,
        inputs.french_gdf,
    )

    # Match user trips to existing locations then parse other columns
    user_journeys_df = _handle_null_values(inputs.raw_journeys_df)
    user_journeys_df = match_loc_ids(user_journeys_df, utils.gdf_to_polars(locations_gdf), inputs.gtfs.stops_df)
    user_journeys_df = user_journeys_df.select(
        user_id=pl.col("id_utilisateur").cast(pl.String),
        journey_id=pl.col("id_deplacement").cast(pl.String),
        leg_id=pl.col("id_trajet").cast(pl.Int8),
        leg_mode=pl.col("mode").cast(pl.Categorical).replace(MODE_MAPPING),
        leg_line=pl.col("ligne_trajet").cast(pl.String),
        duration=pl.duration(microseconds=0),
        dep_day=pl.col("jour_depart").str.to_date("%+"),
        dep_time=pl.col("date").str.split(" - ").list.first().str.to_time("%R"),
        dep_purpose=pl.col("motif_depart").replace_strict(PURPOSE_MAPPING).cast(pl.Categorical),
        dep_loc_id="dep_loc_id",
        arr_loc_id="arr_loc_id",
        arr_purpose=pl.col("motif_arrivee").replace_strict(PURPOSE_MAPPING).cast(pl.Categorical),
    )
    user_journeys_df = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)

    # Filter out locations not in user journeys
    user_loc_ids = (
        pl
        .concat([user_journeys_df["dep_loc_id"], user_journeys_df["arr_loc_id"]])
        .unique()
        .rename("loc_id")
        .to_pandas()
    )
    locations_gdf = locations_gdf.merge(user_loc_ids, on="loc_id")
    locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

    return GenevaData(inputs, locations_gdf, user_journeys_df)


def build_geneva_locations(
    stops: pl.DataFrame,
    subsectors: gpd.GeoDataFrame,
    postcodes: gpd.GeoDataFrame,
    localities: gpd.GeoDataFrame,
    swiss_boundaries: gpd.GeoDataFrame,
    french_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Concatenate all Geneva location types (NA sentinel, PT stops, subsectors, Swiss and French municipalities)."""
    special_locations = build_special_locations()
    pt_locations = _build_pt_locations(stops)
    subsector_locations = _build_subsector_locations(subsectors)
    municipality_swiss_locations = _build_municipality_swiss_locations(postcodes, localities, swiss_boundaries)
    municipality_french_locations = _build_municipality_french_locations(french_gdf)

    return gpd.GeoDataFrame(
        pd.concat(
            [
                special_locations,
                pt_locations,
                subsector_locations,
                municipality_swiss_locations,
                municipality_french_locations,
            ],
            ignore_index=True,
        ),
        crs=CRS,
    )


def _build_pt_locations(stops: pl.DataFrame) -> gpd.GeoDataFrame:
    pt_locations = stops.select(
        loc_id="stop_id", loc_name="stop_name", type=pl.lit("public_transport"), lon="stop_lon", lat="stop_lat"
    )

    return gpd.GeoDataFrame(
        pt_locations.to_pandas(), geometry=gpd.points_from_xy(pt_locations["lon"], pt_locations["lat"], crs=CRS)
    )[LOCATIONS_COLUMNS]


def _build_subsector_locations(subsectors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    subsector_locations = add_lon_lat_from_centroid(subsectors_gdf, index_col="OBJECTID")

    subsector_locations["loc_id"] = "subsector-" + subsector_locations["OBJECTID"].astype(str)
    subsector_locations["type"] = "subsector"

    subsector_locations = subsector_locations.rename(columns={"NOM": "loc_name"})
    return subsector_locations[LOCATIONS_COLUMNS]


def _build_municipality_swiss_locations(
    postcodes_gdf: gpd.GeoDataFrame, localities_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    projected_crs = localities_gdf.estimate_utm_crs()
    localities_gdf = localities_gdf.copy().to_crs(projected_crs)

    geneva_boundaries_gdf = boundaries_gdf[boundaries_gdf["KANTONSNUM"] == GENEVA_CANTON_CODE]
    geneva_shape = geneva_boundaries_gdf.geometry.to_crs(projected_crs).union_all()
    geneva_locality_names = list(geneva_boundaries_gdf["NAME"]) + [
        "Athenaz (Avusy)",
        "La Croix-de-Rozon",
        "Collex",
        "Grand-Lancy",
        "Petit-Lancy",
        "Perly",
        "Chambésy",
        "Carouge GE",
        "Corsier GE",
    ]
    localities_gdf["in_geneva"] = localities_gdf["NAME"].isin(geneva_locality_names)
    localities_gdf["in_geneva"] = localities_gdf["in_geneva"] | localities_gdf.within(geneva_shape)

    postcodes_gdf = postcodes_gdf[["FK_LOCALIT", "ZIP_ID", "ZIP4", "geometry"]]
    postcodes_gdf = add_lon_lat_from_centroid(postcodes_gdf, index_col="ZIP_ID")
    municipality_locations = localities_gdf.drop(columns=["geometry"]).merge(
        postcodes_gdf, left_on="LOCALITYID", right_on="FK_LOCALIT"
    )

    municipality_locations["loc_name"] = municipality_locations["NAME"] + " - " + municipality_locations["ZIP4"]
    municipality_locations["loc_id"] = (
        "CH-" + municipality_locations["ZIP4"] + "-" + municipality_locations["INDEXNAME"]
    )

    in_geneva = municipality_locations["in_geneva"]
    municipality_locations.loc[~in_geneva, "type"] = "municipality_swiss"
    municipality_locations.loc[in_geneva, "type"] = "municipality_geneva"

    municipality_locations = municipality_locations.drop_duplicates(subset="loc_id", keep="first").reset_index()
    return municipality_locations[LOCATIONS_COLUMNS]


def _build_municipality_french_locations(french_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    french_locations = add_lon_lat_from_centroid(french_gdf, index_col="ID")
    french_locations["loc_id"] = "FR-" + french_locations["ID"]
    french_locations["loc_name"] = french_locations["LIB"].str.upper() + " - " + french_locations["ID"]
    french_locations["type"] = "municipality_french"

    return french_locations[LOCATIONS_COLUMNS]


def build_stop_names_to_loc_id_mapping(stops_df: pl.DataFrame) -> pl.DataFrame:
    """Return a ``(loc_name, loc_id)`` DataFrame mapping stop names to their parent station IDs."""
    df = stops_df.with_columns(
        pl
        .when(pl.col("stop_id").str.starts_with("Parent"))
        .then("stop_id")
        .otherwise("parent_station")
        .alias("parent_station")
    )

    with_parents = (
        df
        .filter(pl.col("parent_station") != "")
        .group_by("parent_station")
        .agg(pl.col("stop_name").first().alias("loc_name"))
        .rename({"parent_station": "loc_id"})
        .select("loc_name", "loc_id")
    )

    without_parents = df.filter(pl.col("parent_station") == "", ~pl.col("stop_id").str.starts_with("Parent")).select(
        pl.col("stop_name").alias("loc_name"),
        pl.col("stop_id").alias("loc_id"),
    )

    return pl.concat([with_parents, without_parents])


def match_loc_ids(user_journeys_df: pl.DataFrame, locations_df: pl.DataFrame, stops_df: pl.DataFrame) -> pl.DataFrame:
    """Match raw location strings in journey records to ``loc_id`` values via a cascade of matching strategies."""
    stop_names_to_id_df = build_stop_names_to_loc_id_mapping(stops_df)

    user_journeys_df = manual_patch_stop_names(user_journeys_df, "lieu_depart_trajet", "lieu_arrivee_trajet")
    matched_df = match_loc_id_on_strict_stop_name(user_journeys_df, stop_names_to_id_df)

    columns = [
        ("dep_loc_id", "lieu_depart_trajet", "dep_match_type"),
        ("arr_loc_id", "lieu_arrivee_trajet", "arr_match_type"),
    ]

    for loc_id_col, stop_name_col, match_type_col in columns:
        matched_df = (
            matched_df
            .pipe(match_loc_id_on_subsector, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_swiss, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_french, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_na, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_fuzzy_stop_names, stop_names_to_id_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_null_loc_ids_to_na, loc_id_col, match_type_col)
        )

    return matched_df


def manual_patch_stop_names(user_journeys_df: pl.DataFrame, *stop_name_cols: str) -> pl.DataFrame:
    return user_journeys_df.with_columns(pl.col(stop_name_cols).replace(STOP_NAME_MAPPING))


def match_loc_id_on_strict_stop_name(user_journeys_df: pl.DataFrame, stop_names_to_id_df: pl.DataFrame) -> pl.DataFrame:
    return (
        user_journeys_df
        .join(
            stop_names_to_id_df.select(pl.all().name.prefix("dep_"), dep_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_depart_trajet").str.to_lowercase(),
            right_on=pl.col("dep_loc_name").str.to_lowercase(),
            how="left",
        )
        .join(
            stop_names_to_id_df.select(pl.all().name.prefix("arr_"), arr_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_arrivee_trajet").str.to_lowercase(),
            right_on=pl.col("arr_loc_name").str.to_lowercase(),
            how="left",
        )
        .drop(cs.ends_with("loc_name"))
    )


def match_loc_id_on_subsector(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    regex = LOCATION_REGEXES["subsector"]
    subsectors = all_locations_df.filter(pl.col("type") == "subsector").select("loc_id", "loc_name")
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        user_journeys_df
        .with_columns(pl.col(stop_name_col).str.extract(regex, 1).alias("match"))
        .join(
            subsectors,
            left_on=pl.col("match").str.strip_chars().str.to_lowercase(),
            right_on=pl.col("loc_name").str.to_lowercase(),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr).then(pl.lit("subsector")).otherwise(match_type_col).alias(match_type_col),
        )
        .drop("match", "loc_id", "loc_name")
    )


def match_loc_id_on_municipality_swiss(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    regex = LOCATION_REGEXES["municipality_swiss"]
    municipalities = all_locations_df.filter(
        pl.col("type").is_in(["municipality_swiss", "municipality_geneva"])
    ).select("loc_id", "loc_name")
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        user_journeys_df
        .join(
            municipalities,
            left_on=stop_name_col,
            right_on="loc_name",
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(does_regex_match_expr)
            .then(pl.lit("municipality_swiss"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_loc_id_on_municipality_french(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    regex = LOCATION_REGEXES["municipality_french"]
    municipalities = all_locations_df.filter(pl.col("type") == "municipality_french").select("loc_id", "loc_name")
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        user_journeys_df
        .join(
            municipalities,
            left_on=pl.col(stop_name_col).str.extract(regex, 2),
            right_on=pl.col("loc_id").str.strip_prefix("FR-"),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(does_regex_match_expr)
            .then(pl.lit("municipality_french"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id", "loc_name")
    )


def match_loc_id_on_na(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
) -> pl.DataFrame:
    regex = LOCATION_REGEXES["na"]
    municipalities = all_locations_df.filter(pl.col("type") == "na").select("loc_id", "loc_name")
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        user_journeys_df
        .join(
            municipalities,
            left_on=pl.col(stop_name_col),
            right_on=pl.col("loc_name"),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr).then(pl.lit("na")).otherwise(match_type_col).alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_loc_id_on_fuzzy_stop_names(
    user_journeys_df: pl.DataFrame,
    stop_names_to_id_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
) -> pl.DataFrame:
    unmatched_stop_names = pl.concat([
        user_journeys_df.filter(pl.col("dep_loc_id").is_null())["lieu_depart_trajet"].rename("stop_name"),
        user_journeys_df.filter(pl.col("arr_loc_id").is_null())["lieu_arrivee_trajet"].rename("stop_name"),
    ]).unique()
    unmatched_stop_names = pl.DataFrame(unmatched_stop_names)

    stop_names_list: list[str] = stop_names_to_id_df["loc_name"].str.to_lowercase().to_list()
    stop_ids_list = stop_names_to_id_df["loc_id"].to_list()

    def fuzzy_match(stop_name: str):
        # noinspection PyTypeChecker
        best_name, best_score, best_idx = process.extractOne(stop_name, stop_names_list, scorer=fuzz.ratio)

        return {
            "closest_stop_name": best_name,
            "closest_stop_id": stop_ids_list[best_idx],
            "score": best_score,
        }

    stripped_stop_names = pl.col("stop_name").str.to_lowercase().str.strip_chars("0123456789- ")
    matched_stop_names = (
        unmatched_stop_names
        .with_columns(stripped_stop_names.map_elements(fuzzy_match).alias("result"))
        .unnest("result")
        .filter(pl.col("score") > FUZZY_MATCH_THRESHOLD)
        .select("stop_name", pl.col("closest_stop_id").alias("loc_id"))
    )

    is_fuzzy_match_success = pl.col(loc_id_col).is_null() & pl.col("loc_id").is_not_null()

    return (
        user_journeys_df
        .join(matched_stop_names, left_on=stop_name_col, right_on="stop_name", how="left")
        .with_columns(
            pl.when(is_fuzzy_match_success).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(is_fuzzy_match_success)
            .then(pl.lit("fuzzy_stop_name"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_null_loc_ids_to_na(user_journeys_df: pl.DataFrame, loc_id_col: str, match_type_col: str) -> pl.DataFrame:
    is_unmatched = pl.col(loc_id_col).is_null()

    return user_journeys_df.with_columns(
        pl.when(is_unmatched).then(pl.lit("NA")).otherwise(loc_id_col).alias(loc_id_col),
        pl.when(is_unmatched).then(pl.lit("na")).otherwise(match_type_col).alias(match_type_col),
    )


def _read_raw_data(data_cfg: DataConfig, project_root=None) -> pl.DataFrame:
    project_root = project_root if project_root is not None else Path(".")
    data_path = project_root / data_cfg.paths.raw

    raw_geneva_df = utils.read_from_parquet(
        data_path / data_cfg.inputs.raw_journeys,
        schema={
            "id_utilisateur": pl.String,
            "id_deplacement": pl.Categorical,
            "id_trajet": pl.Int8,
            "jour_depart": pl.String,
            "motif_depart": pl.Categorical,
            "motif_arrivee": pl.Categorical,
            "date": pl.String,
            "lieu_depart_trajet": pl.String,
            "lieu_arrivee_trajet": pl.String,
            "mode": pl.Categorical,
            "ligne_trajet": pl.String,
        },
    )

    raw_geneva_df = raw_geneva_df.with_columns(pl.col("jour_depart").str.to_date(format="%+", strict=False))

    return raw_geneva_df


def _handle_null_values(raw_geneva_df: pl.DataFrame) -> pl.DataFrame:
    # Remove null rows
    cols_remove_null_rows = ["jour_depart"]
    removed_rows_df = raw_geneva_df.drop_nulls(subset=cols_remove_null_rows)

    # Remove empty rows from certain columns
    cols_remove_empty_rows = ["jour_depart", "date", "motif_depart", "motif_arrivee"]
    removed_rows_df = removed_rows_df.filter(pl.all_horizontal(pl.col(cols_remove_empty_rows) != ""))

    # Impute empty rows to `NA` category for location columns
    cols_impute_empty_to_na = ["lieu_depart_trajet", "lieu_arrivee_trajet"]
    imputed_na_rows_df = removed_rows_df.with_columns(pl.col(cols_impute_empty_to_na).replace(old="", new="NA"))

    # Impute empty values to unknown for mode column
    cols_impute_empty_to_unknown = ["mode"]
    imputed_unknown_rows_df = imputed_na_rows_df.with_columns(
        pl.col(cols_impute_empty_to_unknown).replace(old="", new=Mode.UNKNOWN)
    )

    # Impute empty values of `ligne_trajet` to UNKNOWN or NA for ligne column depending on if mode is applicable
    applicable_modes = ["mode_bus", "mode_tramway", "mode_bateau_navette"]
    imputed_line_df = imputed_unknown_rows_df.with_columns(
        pl
        .when((pl.col("ligne_trajet") == "") & pl.col("mode").is_in(applicable_modes))
        .then(pl.lit("UNKNOWN"))
        .otherwise("ligne_trajet")
        .alias("ligne_trajet")
    ).with_columns(
        pl.when(pl.col("ligne_trajet") == "").then(pl.lit("NA")).otherwise("ligne_trajet").alias("ligne_trajet")
    )

    return imputed_line_df
