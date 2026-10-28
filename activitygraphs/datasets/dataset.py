"""This module contains VisitDataset, the ``load_dataset`` cache entry point, and the split/scaler cache fingerprints."""

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from functools import cached_property, partial
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
import torch_geometric as pyg
from omegaconf import OmegaConf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from activitygraphs.config import Config, DataConfig
from activitygraphs.graph import load_graph
from activitygraphs.schema import DatasetSchema
from activitygraphs.surveys.registry import load_survey
from activitygraphs.tensors import SurveyTensors, build_tensors
from activitygraphs.utils import get_project_root

type Fingerprint = dict


class ProcessedFile(StrEnum):
    """Names of the files cached by ``VisitDataset`` in its ``processed`` directory."""

    PYG_GRAPH = "pyg_graph.pt"
    USER_NODE_FEATURES = "user_node_features.pt"
    VISIT_LABELS = "visit_labels.pt"
    USER_ATTRIBUTES = "user_attributes.pt"
    DISTANCES = "distances.pt"
    GROUP_IDS = "group_ids.pt"
    USER_IDS = "user_ids.parquet"
    NODE_IDS = "node_ids.parquet"
    SCHEMA = "schema.json"
    FINGERPRINT = "fingerprint.json"


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


class VisitDataset(pyg.data.Dataset):
    """Main PyG dataset for a travel survey: shared graph + per-user node features/visit labels + user attributes.

    Stores the graph once and slices per-user tensors on ``get()`` call.
    Each item returned by ``get(i)`` is a ``pyg.data.Data`` with additional tensors:
      - ``x_loc = location_features``,
      - ``x_user_node = user_node_features[i]``,
      - ``y = visit_labels[i]``,
      - ``user_attr = user_attributes[i]``,
      - ``distances = distances[i]``, the model feature, scaled once the scalers are applied, and
      - ``distances_m = distances_m[i]``, the same distances in metres, never scaled.
    """

    PROCESSED_FILE_NAMES = [str(file) for file in ProcessedFile]

    def __init__(
        self,
        root: str,
        build: Callable[[], SurveyTensors] | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        """Load the cached tensors under ``root``, building and caching them first if any is missing.

        Args:
            root: Dataset directory; the tensors are cached in the ``processed`` subdirectory.
            build: Returns the survey tensors, called only when the cache is incomplete. ``None`` for a caller that
                only reads the cache.
            transform: PyG transform applied to each item.
            pre_transform: PyG transform applied once to the shared graph before caching.
            pre_filter: PyG pre-filter, unused.

        Raises:
            ValueError: The cache is incomplete and no ``build`` is given.
        """
        self._build = build

        super().__init__(root, transform, pre_transform, pre_filter)

        processed_dir = Path(self.processed_dir)
        self.pyg_graph: pyg.data.Data = torch.load(processed_dir / ProcessedFile.PYG_GRAPH, weights_only=False)
        self.user_node_features: torch.Tensor = torch.load(
            processed_dir / ProcessedFile.USER_NODE_FEATURES, weights_only=True
        )
        self.visit_labels: torch.Tensor = torch.load(processed_dir / ProcessedFile.VISIT_LABELS, weights_only=True)
        self.user_attributes: torch.Tensor = torch.load(
            processed_dir / ProcessedFile.USER_ATTRIBUTES, weights_only=True
        )
        self.distances_m: torch.Tensor = torch.load(processed_dir / ProcessedFile.DISTANCES, weights_only=True)
        self.distances: torch.Tensor = self.distances_m
        self.group_ids: torch.Tensor = torch.load(processed_dir / ProcessedFile.GROUP_IDS, weights_only=True)
        self.user_ids: pl.DataFrame = pl.read_parquet(processed_dir / ProcessedFile.USER_IDS)
        self.node_ids: pl.DataFrame = pl.read_parquet(processed_dir / ProcessedFile.NODE_IDS)
        self.schema: DatasetSchema = DatasetSchema.read(processed_dir / ProcessedFile.SCHEMA)

        # Check that each user has one and only one home node
        is_home_idx = self.schema.user_nodes.index_of("is_home")
        home_counts = self.user_node_features[:, :, is_home_idx].sum(dim=1)
        if not torch.all(home_counts == 1):
            num_bad = int((home_counts != 1).sum())
            raise ValueError(f"{num_bad} users do not have exactly one home node. Expected one per user.")

        if self.pyg_graph.x is None or self.pyg_graph.edge_index is None or self.pyg_graph.edge_attr is None:
            raise ValueError("Graph must have node features, an edge index and edge features.")

        # Check that the graph actually has nodes
        num_nodes = self.pyg_graph.num_nodes
        if num_nodes is None:
            raise ValueError("Graph has `None` number of nodes")

        self.num_users: int = self.user_attributes.size(0)
        self.num_nodes: int = num_nodes
        self._is_scaled: bool = False

    @property
    def location_x(self) -> torch.Tensor:
        """Graph node features, ``[num_nodes, num_location_features]``."""
        return cast(torch.Tensor, self.pyg_graph.x)

    @property
    def edge_index(self) -> torch.Tensor:
        """Graph edge list, ``[2, num_edges]``."""
        return cast(torch.Tensor, self.pyg_graph.edge_index)

    @property
    def edge_attr(self) -> torch.Tensor:
        """Graph edge features, ``[num_edges, num_edge_features]``."""
        return cast(torch.Tensor, self.pyg_graph.edge_attr)

    @property
    def num_rwpe_features(self) -> int:
        """Number of columns of the random-walk positional encoding, ``random_walk_pe``."""
        return self.pyg_graph.random_walk_pe.shape[1]

    @cached_property
    def home_hop_distance(self) -> np.ndarray:
        """All-pairs hop distance matrix (over spatial adjacency graph) ``[num_nodes, num_nodes]`` for the graph."""
        return compute_home_hop_distance(self.edge_index, self.num_nodes)

    @cached_property
    def median_realised_size(self) -> int:
        """Median number of visited nodes per user over the dataset, counting the home node."""
        sizes = [int(data.y.sum()) for data in self]
        return int(np.median(sizes))

    @property
    def raw_file_names(self) -> list[str]:
        """No raw files: the dataset is built from in-memory tensors."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Names of the cached tensor files."""
        return self.PROCESSED_FILE_NAMES

    def download(self) -> None:
        """Do nothing: there is nothing to download."""
        pass

    def process(self):
        """Build the survey tensors, apply the pre-transform to the graph and write everything to the cache."""
        if self._build is None:
            raise ValueError(f"The tensors at `{self.processed_dir}` are not cached and no `build` was given.")

        tensors = self._build()
        graph = tensors.pyg_graph
        unf = tensors.user_node_features
        vl = tensors.visit_labels
        attr = tensors.user_attributes
        dist = tensors.distances
        group_ids = tensors.group_ids

        num_unf_users, num_unf_nodes, _ = unf.shape
        num_vl_users, num_vl_nodes, _ = vl.shape
        num_attr_users, _ = attr.shape
        num_di_users, num_di_nodes, _ = dist.shape
        num_grp_users = group_ids.shape[0]

        if not (graph.num_nodes == num_unf_nodes == num_vl_nodes == num_di_nodes):
            raise ValueError(
                f"Node count mismatch: graph has {graph.num_nodes}, user_node_features has {num_unf_nodes}, "
                f"visit_labels has {num_vl_nodes}, distances has {num_di_nodes}."
            )
        if not (num_unf_users == num_vl_users == num_attr_users == num_di_users == num_grp_users):
            raise ValueError(
                f"User count mismatch: user_node_features has {num_unf_users}, visit_labels has {num_vl_users}, "
                f"user_attributes has {num_attr_users}, distances has {num_di_users}."
            )

        tensor_widths = {
            "locations": graph.num_node_features,
            "edges": graph.num_edge_features,
            "user_nodes": unf.shape[-1],
            "user_attributes": attr.shape[-1],
            "distances": dist.shape[-1],
            "targets": vl.shape[-1],
        }

        for group_schema in tensors.schema.groups:
            width = tensor_widths[group_schema.group]
            if width != group_schema.width:
                raise ValueError(
                    f"The `{group_schema.group}` tensor has {width} columns but its schema has {group_schema.width}."
                )

        if tensors.user_ids.height != num_unf_users or tensors.node_ids.height != num_unf_nodes:
            raise ValueError(
                f"Id count mismatch: {tensors.user_ids.height} user ids for {num_unf_users} users, "
                f"{tensors.node_ids.height} node ids for {num_unf_nodes} nodes."
            )

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        processed_dir = Path(self.processed_dir)
        torch.save(graph, processed_dir / ProcessedFile.PYG_GRAPH)
        torch.save(unf, processed_dir / ProcessedFile.USER_NODE_FEATURES)
        torch.save(vl, processed_dir / ProcessedFile.VISIT_LABELS)
        torch.save(attr, processed_dir / ProcessedFile.USER_ATTRIBUTES)
        torch.save(dist, processed_dir / ProcessedFile.DISTANCES)
        torch.save(group_ids, processed_dir / ProcessedFile.GROUP_IDS)
        tensors.user_ids.write_parquet(processed_dir / ProcessedFile.USER_IDS)
        tensors.node_ids.write_parquet(processed_dir / ProcessedFile.NODE_IDS)
        tensors.schema.save(processed_dir / ProcessedFile.SCHEMA)
        (processed_dir / ProcessedFile.FINGERPRINT).write_text(json.dumps(tensors_fingerprint(tensors), indent=2))

    def len(self) -> int:
        """Return the number of users."""
        return self.num_users

    def get(self, idx: int) -> pyg.data.Data:
        """Return user ``idx``'s graph: shared location features, and user's node features and labels."""
        if not 0 <= idx < self.num_users:
            raise IndexError(f"Index {idx} out of range for {self.num_users} users.")

        location_x = self.location_x
        user_node_x = self.user_node_features[idx]

        # Cast user node features to same type as graph features if mismatch
        if user_node_x.dtype != location_x.dtype:
            user_node_x = user_node_x.to(location_x.dtype)

        # Create new user-annotated graph
        y = self.visit_labels[idx]
        user_attr = self.user_attributes[idx]
        edge_index = self.edge_index
        edge_attr = self.edge_attr

        data = pyg.data.Data(
            x_loc=location_x,
            x_user_node=user_node_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=self.num_nodes,
        )

        # Unsqueeze so batching works
        data.user_attr = user_attr.unsqueeze(0)
        data.user_id = torch.tensor([idx], dtype=torch.long)

        # Per-user distance-from-home
        data.distances = self.distances[idx].to(location_x.dtype)
        data.distances_m = self.distances_m[idx]

        # Carry over any extra attributes from the graph (pos, etc.)
        for key, value in self.pyg_graph:  # pyrefly: ignore[not-iterable]
            if key in ("x", "edge_index", "edge_attr", "y"):
                continue

            data[key] = value

        return data


def compute_home_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return the all-pairs contiguity-hop distance matrix ``[num_nodes, num_nodes]``.

    Treats the graph as undirected and unweighted; unreachable pairs are ``inf``.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))

    return shortest_path(adjacency, method="D", unweighted=True, directed=False)


def build_survey_tensors(cfg: DataConfig, project_root: Path) -> SurveyTensors:
    """Return the tensors of the survey and graph described by ``cfg``, loading or building their caches."""
    survey = load_survey(cfg, project_root).with_filter("zone")
    nodes_gdf, edges_gdf = load_graph(survey, cfg, project_root)

    return build_tensors(survey, nodes_gdf, edges_gdf)


def load_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **dataset_kwargs,
) -> VisitDataset:
    """Return the ``VisitDataset`` rooted at ``cfg.data.paths.tensors``, building its cache from the survey if needed."""
    project_root = get_project_root(project_root)
    build = partial(build_survey_tensors, cfg.data, project_root)

    return VisitDataset(str(project_root / cfg.data.paths.tensors), build, **dataset_kwargs)


# =========================================
# Fingerprints
# =========================================


def tensors_fingerprint(tensors: SurveyTensors) -> Fingerprint:
    """Return the identity of ``tensors`` stored beside the cache: their shapes and hashes of the ids and the schema."""
    ids = tensors.user_ids.write_csv() + tensors.node_ids.write_csv()
    schema = json.dumps(tensors.schema.to_dict(), sort_keys=True)

    return {
        "n_users": tensors.user_attributes.shape[0],
        "n_nodes": tensors.user_node_features.shape[1],
        "n_user_node_cols": tensors.user_node_features.shape[-1],
        "n_user_attribute_cols": tensors.user_attributes.shape[-1],
        "n_groups": len(tensors.group_ids.unique()),
        "ids": hashlib.blake2b(ids.encode(), digest_size=16).hexdigest(),
        "schema": hashlib.blake2b(schema.encode(), digest_size=16).hexdigest(),
    }


def dataset_fingerprint(cfg: Config, tensors_fp: Fingerprint) -> Fingerprint:
    """Return the fingerprint the split and scaler caches are checked against: ``tensors_fp`` and the data config.

    Only ``cfg.data`` is hashed, so training hyperparameters do not invalidate the caches.
    """
    config_str = OmegaConf.to_yaml(cfg.data, resolve=True, sort_keys=True)

    return tensors_fp | {"data_config": hashlib.blake2b(config_str.encode(), digest_size=16).hexdigest()}


def read_fingerprint(cfg: Config, project_root: Path) -> Fingerprint:
    """Return the ``dataset_fingerprint`` of the cached tensors of ``cfg``, without loading them.

    Raises:
        FileNotFoundError: The tensors are not cached.
    """
    path = project_root / cfg.data.paths.tensors / "processed" / ProcessedFile.FINGERPRINT

    return dataset_fingerprint(cfg, json.loads(path.read_text()))
