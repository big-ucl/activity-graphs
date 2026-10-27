"""This module contains VisitDataset, the ``load_dataset`` cache entry point, and the split/scaler cache fingerprints."""

import hashlib
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch_geometric as pyg
from omegaconf import OmegaConf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from activitygraphs.config import Config
from activitygraphs.graph import load_graph
from activitygraphs.surveys.registry import load_survey
from activitygraphs.tensors import USER_NODE_FEATURE_NAMES, SurveyTensors, build_tensors
from activitygraphs.utils import get_project_root

type Fingerprint = dict


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


class VisitDataset(pyg.data.Dataset):
    """Main PyG dataset for a travel survey: shared graph + per-user node features/visit labels + user attributes.

    Stores the graph once and slices per-user tensors on ``get()`` call.
    Each item returned by ``get(i)`` is a ``pyg.data.Data`` with ``x = [location_features || user_node_features[i]]``,
    ``y = visit_labels[i]``, and ``user_attr = user_attributes[i]``.
    """

    PROCESSED_FILE_NAMES = [
        "pyg_graph.pt",
        "user_node_features.pt",
        "visit_labels.pt",
        "user_attributes.pt",
        "distances.pt",
        "group_ids.pt",
    ]

    def __init__(
        self,
        root: str,
        tensors: SurveyTensors | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._tensors_in = tensors

        super().__init__(root, transform, pre_transform, pre_filter)
        del self._tensors_in  # pyrefly: ignore[unsupported-delete]

        processed_dir = Path(self.processed_dir)
        self.pyg_graph: pyg.data.Data = torch.load(processed_dir / "pyg_graph.pt", weights_only=False)
        self.user_node_features: torch.Tensor = torch.load(processed_dir / "user_node_features.pt", weights_only=True)
        self.visit_labels: torch.Tensor = torch.load(processed_dir / "visit_labels.pt", weights_only=True)
        self.user_attributes: torch.Tensor = torch.load(processed_dir / "user_attributes.pt", weights_only=True)
        self.distances: torch.Tensor = torch.load(processed_dir / "distances.pt", weights_only=True)
        self.group_ids: torch.Tensor = torch.load(processed_dir / "group_ids.pt", weights_only=True)

        # Check that each user has one and only one home node
        home_counts = self.user_node_features[:, :, self.is_home_user_node_idx].sum(dim=1)
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
    def num_location_features(self) -> int:
        """Number of location node-feature columns (those preceding the user node block in ``x``)."""
        return self.location_x.shape[1]

    @property
    def num_rwpe_features(self) -> int:
        """Number of columns of the random-walk positional encoding, ``random_walk_pe``."""
        return self.pyg_graph.random_walk_pe.shape[1]

    @property
    def is_home_user_node_idx(self) -> int:
        """Column index of ``is_home`` within the per-user node feature block."""
        return USER_NODE_FEATURE_NAMES.index("is_home")

    @property
    def is_home_col_idx(self) -> int:
        """Column index of ``is_home`` within the concatenated node feature matrix ``x``."""
        return self.num_location_features + self.is_home_user_node_idx

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
        """Apply the pre-transform to the graph and write all tensors to the cache."""
        if self._tensors_in is None:
            raise ValueError("First-time construction requires the survey tensors to be passed to init.")

        graph = self._tensors_in.pyg_graph
        unf = self._tensors_in.user_node_features
        vl = self._tensors_in.visit_labels
        attr = self._tensors_in.user_attributes
        dist = self._tensors_in.distances
        group_ids = self._tensors_in.group_ids

        num_unf_users, num_unf_nodes, _ = unf.shape
        num_vl_users, num_vl_nodes, num_vl_labels = vl.shape
        num_attr_users, _ = attr.shape
        num_di_users, num_di_nodes, num_di_feat = dist.shape
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
        if num_vl_labels != 1:
            raise ValueError(f"visit_labels last dim must be 1, got {num_vl_labels}.")
        if num_di_feat != 1:
            raise ValueError(f"distances last dim must be 1, got {num_di_feat}.")

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        processed_dir = Path(self.processed_dir)
        torch.save(graph, processed_dir / "pyg_graph.pt")
        torch.save(unf, processed_dir / "user_node_features.pt")
        torch.save(vl, processed_dir / "visit_labels.pt")
        torch.save(attr, processed_dir / "user_attributes.pt")
        torch.save(dist, processed_dir / "distances.pt")
        torch.save(group_ids, processed_dir / "group_ids.pt")

    def len(self) -> int:
        """Return the number of users."""
        return self.num_users

    def get(self, idx: int) -> pyg.data.Data:
        """Return user ``idx``'s graph: shared location features joined with the user's features and labels."""
        if not 0 <= idx < self.num_users:
            raise IndexError(f"Index {idx} out of range for {self.num_users} users.")

        location_x = self.location_x
        user_node_x = self.user_node_features[idx]

        # Cast user node features to same type as graph features if mismatch
        if user_node_x.dtype != location_x.dtype:
            user_node_x = user_node_x.to(location_x.dtype)

        # Create new user-annotated graph
        full_x = torch.cat([location_x, user_node_x], dim=1)
        y = self.visit_labels[idx]
        user_attr = self.user_attributes[idx]
        edge_index = self.edge_index
        edge_attr = self.edge_attr

        data = pyg.data.Data(x=full_x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Unsqueeze so batching works
        data.user_attr = user_attr.unsqueeze(0)
        data.user_id = torch.tensor([idx], dtype=torch.long)

        # Per-user distance-from-home
        data.distances = self.distances[idx].to(full_x.dtype)

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


def load_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **build_kwargs,
) -> VisitDataset:
    """Construct an ``VisitDataset`` rooted at ``cfg.data.paths.tensors``."""
    project_root = get_project_root(project_root)
    dataset_path = project_root / cfg.data.paths.tensors

    if not all((dataset_path / file).exists() for file in VisitDataset.PROCESSED_FILE_NAMES):
        survey = load_survey(cfg.data, project_root).with_filter("zone")
        nodes_gdf, edges_gdf = load_graph(survey, cfg.data, project_root)
        tensors = build_tensors(survey, nodes_gdf, edges_gdf)

        return VisitDataset(str(dataset_path), tensors, **build_kwargs)

    return VisitDataset(root=str(dataset_path), **build_kwargs)


def dataset_fingerprint(cfg: Config, dataset: VisitDataset) -> Fingerprint:
    """Returns a fingerprint of the dataset for comparison with caches, based on the data config and dataset properties.

    Only ``cfg.data`` is hashed, so training hyperparameters do not invalidate the caches. Must be
    called before ``apply_scalers``, which rewrites the feature tensors in place.
    """
    config_str = OmegaConf.to_yaml(cfg.data, resolve=True, sort_keys=True)

    return {
        "n_users": len(dataset),
        "n_nodes": dataset.num_nodes,
        "n_user_node_cols": int(dataset.user_node_features.shape[-1]),
        "n_user_attribute_cols": int(dataset.user_attributes.shape[-1]),
        "n_groups": len(dataset.group_ids.unique()),
        "data_config": hashlib.blake2b(config_str.encode(), digest_size=16).hexdigest(),
    }
