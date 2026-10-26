"""ActivityDataset, FittedScalers, and the ``load_splits`` entry point for the ML pipeline."""

import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from activitygraphs.config import Config
from activitygraphs.graph import load_graph
from activitygraphs.ml.metrics import compute_home_hop_distance
from activitygraphs.surveys.registry import load_survey
from activitygraphs.tensors import USER_NODE_FEATURE_NAMES, SurveyTensors, build_tensors
from activitygraphs.utils import get_project_root

type Fingerprint = dict


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


class ActivityDataset(pyg.data.Dataset):
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


@dataclass
class FittedScalers:
    """Container for ``StandardScaler`` instances fitted on the training split."""

    node_features: StandardScaler
    edge_features: StandardScaler
    user_node_features: StandardScaler | None
    user_attributes: StandardScaler
    distances: StandardScaler
    exclude_user_node_cols: list[int]
    fingerprint: Fingerprint

    def save(self, path: Path) -> None:
        """Pickle the scalers to ``path``."""
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def read(path: Path) -> "FittedScalers":
        """Read scalers pickled by ``save``."""
        with path.open("rb") as f:
            return pickle.load(f)


def load_splits(
    cfg: Config,
    val_size: float,
    test_size: float,
    seed: int,
    project_root: Path | None = None,
    **build_kwargs,
) -> tuple[ActivityDataset, ActivityDataset, ActivityDataset, FittedScalers]:
    """Load, split, scale, and return the dataset as train/val/test subsets with cached scalers.

    Split indices are cached in ``splits.json``; scalers are cached in ``scalers.pkl``
    under ``cfg.data.paths.pyg_datasets``.

    Returns:
        Tuple ``(train_dataset, val_dataset, test_dataset, scalers)``.
    """
    project_root = get_project_root(project_root)
    pyg_dir = project_root / cfg.data.paths.pyg_datasets
    splits_cache = pyg_dir / f"splits_{seed}_val{val_size}_test{test_size}_grouped.json"
    scalers_cache = pyg_dir / f"scalers_{seed}_val{val_size}_test{test_size}_grouped.pkl"

    positional_encodings_transforms = T.Compose([
        T.AddRandomWalkPE(walk_length=20, attr_name=None),
        # T.AddLaplacianEigenvectorPE(k=8, attr_name=None), # Removed due to instability on different graphs.
    ])

    dataset = load_dataset(
        cfg, project_root=project_root, pre_transform=positional_encodings_transforms, **build_kwargs
    )

    fingerprint = dataset_fingerprint(cfg, dataset)

    train_idx, val_idx, test_idx = split_indices(
        dataset, val_size, test_size, seed, cache_path=splits_cache, fingerprint=fingerprint
    )

    scalers = None
    if scalers_cache.exists():
        scalers = FittedScalers.read(scalers_cache)

    if scalers is None or scalers.fingerprint != fingerprint:
        scalers = fit_scalers(
            dataset, train_idx, exclude_user_node_cols=[dataset.is_home_user_node_idx], fingerprint=fingerprint
        )
        scalers.save(scalers_cache)

    apply_scalers(dataset, scalers)

    return (
        cast(ActivityDataset, dataset[train_idx]),
        cast(ActivityDataset, dataset[val_idx]),
        cast(ActivityDataset, dataset[test_idx]),
        scalers,
    )


def load_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **build_kwargs,
) -> ActivityDataset:
    """Construct an ``ActivityDataset`` rooted at ``cfg.data.paths.pyg_datasets``."""
    project_root = get_project_root(project_root)
    dataset_path = project_root / cfg.data.paths.pyg_datasets

    if not all((dataset_path / file).exists() for file in ActivityDataset.PROCESSED_FILE_NAMES):
        survey = load_survey(cfg.data, project_root).with_filter("zone")
        nodes_gdf, edges_gdf = load_graph(survey, cfg.data, project_root)
        tensors = build_tensors(survey, nodes_gdf, edges_gdf)

        return ActivityDataset(str(dataset_path), tensors, **build_kwargs)

    return ActivityDataset(root=str(dataset_path), **build_kwargs)


def split_indices(
    dataset: ActivityDataset,
    val_size: float,
    test_size: float,
    seed: int,
    fingerprint: Fingerprint,
    cache_path: Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_indices, val_indices, test_indices). Loads from ``cache_path`` if it matches ``fingerprint``, otherwise splits and caches."""
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            cached_indices = json.load(f)

        if cached_indices.get("fingerprint") == fingerprint:
            return cached_indices["train"], cached_indices["val"], cached_indices["test"]

        print(f"Cached indices at `{cache_path.name}` do not match fingerprint, re-splitting.")

    all_idx = list(range(len(dataset)))

    # Split Test and Train-Val
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(iter(gss_test.split(all_idx, groups=dataset.group_ids)))

    # Compute size of validation set and adjust the group Ids
    train_val_group_ids = dataset.group_ids[train_val_idx]
    proportional_val_size = val_size / (1 - test_size)

    # Split Val and Train
    gss_val = GroupShuffleSplit(n_splits=1, test_size=proportional_val_size, random_state=seed)
    rel_train_idx, rel_val_idx = next(iter(gss_val.split(train_val_idx, groups=train_val_group_ids)))
    train_idx, val_idx = train_val_idx[rel_train_idx], train_val_idx[rel_val_idx]

    assert set(train_idx) & set(val_idx) == set() and set(train_val_idx) & set(test_idx) == set(), (
        "Assert train/val/test sets are disjoint."
    )

    train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(
                {"train": train_idx, "val": val_idx, "test": test_idx, "seed": seed, "fingerprint": fingerprint}, f
            )

    return train_idx, val_idx, test_idx


def fit_scalers(
    dataset: ActivityDataset,
    train_idx: list[int],
    fingerprint: Fingerprint,
    exclude_user_node_cols: list[int],
    exclude_user_attribute_cols: list[int] | None = None,
) -> FittedScalers:
    """Fit ``StandardScaler`` instances on the training split of each feature group.

    Args:
        dataset: The full ``ActivityDataset`` (not yet scaled).
        train_idx: Indices of training users.
        exclude_user_node_cols: Column indices excluded from user node scaler fitting (e.g. ``[is_home_user_node_idx]``).
        exclude_user_attribute_cols: Column indices excluded from user attribute scaler fitting.
        fingerprint: Identity of the dataset being fitted on, stored for cache validation.

    Returns:
        ``FittedScalers`` with one scaler per feature group (graph nodes, edges, user nodes, user attributes).
    """
    # Location node features: no need to separate between train and test, the graph is the same.
    node_scaler = StandardScaler()
    node_scaler.fit(dataset.location_x.numpy())

    # Location edge attributes: same as above
    edge_scaler = StandardScaler()
    edge_scaler.fit(dataset.edge_attr.numpy())

    # User node features: user-specific, fit only on train. Skip when every column is excluded (e.g. only is_home).
    user_node_train = dataset.user_node_features[train_idx]
    keep = [c for c in range(user_node_train.shape[-1]) if c not in exclude_user_node_cols]
    user_node_scaler = None
    if keep:
        user_node_flat = user_node_train[..., keep].reshape(-1, len(keep)).float().numpy()
        user_node_scaler = StandardScaler()
        user_node_scaler.fit(user_node_flat)

    # User attributes: user-specific, fit only on train
    attr_train = dataset.user_attributes[train_idx]
    if exclude_user_attribute_cols:
        keep = [c for c in range(attr_train.shape[-1]) if c not in exclude_user_attribute_cols]
        attr_train = attr_train[..., keep]
    attr_scaler = StandardScaler()
    attr_scaler.fit(attr_train.float().numpy())

    # Distance-from-home: user-specific, fit only on train
    distances_train = dataset.distances[train_idx]
    distances_scaler = StandardScaler()
    distances_scaler.fit(distances_train.reshape(-1, distances_train.shape[-1]).float().numpy())

    return FittedScalers(
        node_features=node_scaler,
        edge_features=edge_scaler,
        user_node_features=user_node_scaler,
        user_attributes=attr_scaler,
        distances=distances_scaler,
        exclude_user_node_cols=exclude_user_node_cols,
        fingerprint=fingerprint,
    )


def apply_scalers(dataset: ActivityDataset, scalers: FittedScalers) -> None:
    """Apply fitted scalers to the dataset in-place, transforming graph/user node/user attribute tensors."""
    if dataset._is_scaled:
        raise RuntimeError("apply_scalers called twice on the same ActivityDataset")

    x = dataset.location_x.numpy()
    dataset.pyg_graph.x = torch.from_numpy(scalers.node_features.transform(x)).float()

    ea = dataset.edge_attr.numpy()
    dataset.pyg_graph.edge_attr = torch.from_numpy(scalers.edge_features.transform(ea)).float()

    if scalers.user_node_features is not None:
        unf = dataset.user_node_features.float()
        n_cols = unf.shape[-1]
        keep = [c for c in range(n_cols) if c not in scalers.exclude_user_node_cols]
        flat = unf.reshape(-1, n_cols)[:, keep].numpy()
        scaled_keep = torch.from_numpy(scalers.user_node_features.transform(flat)).float()
        result = unf.reshape(-1, n_cols).clone()
        result[:, keep] = scaled_keep
        dataset.user_node_features = result.reshape(unf.shape)

    attr = dataset.user_attributes.float().numpy()
    dataset.user_attributes = torch.from_numpy(scalers.user_attributes.transform(attr)).float()

    dist = dataset.distances.float()
    shape = dist.shape
    flat = dist.reshape(-1, shape[-1]).numpy()
    dataset.distances = torch.from_numpy(scalers.distances.transform(flat)).float().reshape(shape)

    dataset._is_scaled = True


def dataset_fingerprint(cfg: Config, dataset: ActivityDataset) -> Fingerprint:
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
