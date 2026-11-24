"""Tensor caches of a survey: saves the tensors (in one directory per bundle), the schema, and the fingerprints."""

import hashlib
import json
from pathlib import Path

import polars as pl

from activitygraphs.config import DataConfig
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors.bundles import (
    LocationFile,
    LocationTensors,
    SurveyTensors,
    UserTensors,
    check_tensors,
)

type Fingerprint = dict

TENSORS_DIR = "Tensors"
LOCATIONS_DIR = "locations"
USERS_DIR = "users"
SCHEMA_FILE = "schema.json"
FINGERPRINT_FILE = "fingerprint.json"


def tensors_path(cfg: DataConfig, project_root: Path) -> Path:
    """Return the directory of the tensor cache of the survey of ``cfg``, ``cfg.paths.processed/Tensors/``."""
    return project_root / cfg.paths.processed / TENSORS_DIR


def save_tensors(root: Path, locations: LocationTensors, users: UserTensors, schema: DatasetSchema) -> None:
    """Write the bundles, the schema and the fingerprint of the tensors to the cache under ``root``."""
    locations.save(root / LOCATIONS_DIR)
    users.save(root / USERS_DIR)
    schema.save(root / SCHEMA_FILE)

    fingerprint = _tensors_fingerprint(locations, users, schema)
    (root / FINGERPRINT_FILE).write_text(json.dumps(fingerprint, indent=2))


def read_tensors(root: Path) -> SurveyTensors:
    """Read the bundles and the schema written by ``save_tensors`` to ``root``.

    Raises:
        FileNotFoundError: Any file of the cache is missing.
        ValueError: The cached tensors fail ``check_tensors``.
    """
    _require_cached(root)
    locations = LocationTensors.read(root / LOCATIONS_DIR)
    users = UserTensors.read(root / USERS_DIR)
    schema = DatasetSchema.read(root / SCHEMA_FILE)

    check_tensors(locations, users, schema)

    return locations, users, schema


def read_schema(root: Path) -> DatasetSchema:
    """Read the dataset schema written by ``save_tensors`` to ``root``, without loading the tensors."""
    return DatasetSchema.read(root / SCHEMA_FILE)


def read_node_ids(root: Path) -> pl.DataFrame:
    """Read the survey ``loc_id`` of each node written by ``save_tensors`` to ``root``, without loading the tensors."""
    return pl.read_parquet(root / LOCATIONS_DIR / LocationFile.NODE_IDS)


def read_tensors_fingerprint(root: Path) -> Fingerprint:
    """Return the fingerprint of the tensors cached under ``root``, without loading them.

    Raises:
        FileNotFoundError: Any file of the cache is missing.
    """
    _require_cached(root)

    return json.loads((root / FINGERPRINT_FILE).read_text())


def _tensor_cache_files(root: Path) -> list[Path]:
    """Return the path of every file of the tensor cache under ``root``."""
    location_files = [root / LOCATIONS_DIR / file for file in LocationTensors.file_names()]
    user_files = [root / USERS_DIR / file for file in UserTensors.file_names()]

    return [*location_files, *user_files, root / SCHEMA_FILE, root / FINGERPRINT_FILE]


def _require_cached(root: Path) -> None:
    """Raise unless every file of the tensor cache under ``root`` exists.

    Raises:
        FileNotFoundError: Any file of the cache is missing.
    """
    missing = [str(path.relative_to(root)) for path in _tensor_cache_files(root) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"The tensor cache at `{root}` is missing {missing}.")


def _tensors_fingerprint(locations: LocationTensors, users: UserTensors, schema: DatasetSchema) -> Fingerprint:
    """Return the identity of the tensors stored beside the cache: their shapes and hashes of the ids and the schema."""
    ids = users.user_ids.write_csv() + locations.node_ids.write_csv()
    schema_str = json.dumps(schema.to_dict(), sort_keys=True)

    return {
        "n_users": users.attributes.shape[0],
        "n_nodes": locations.node_ids.height,
        "n_user_attribute_cols": users.attributes.shape[-1],
        "n_groups": len(users.group_ids.unique()),
        "ids": hashlib.blake2b(ids.encode(), digest_size=16).hexdigest(),
        "schema": hashlib.blake2b(schema_str.encode(), digest_size=16).hexdigest(),
    }
