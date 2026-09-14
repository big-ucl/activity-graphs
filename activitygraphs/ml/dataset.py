"""ActivityDataset, FittedScalers, and the ``load_dataset`` entry point for the ML pipeline."""

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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from activitygraphs.config import Config
from activitygraphs.dataprocessing import SPATIAL_FEATURE_NAMES, convert_to_torch, load_data
from activitygraphs.ml.metrics import compute_home_hop_distance
from activitygraphs.utils import get_project_root

# Full-`x` column index of `is_home` in the legacy GenevaDataset pickles only.
# The main ActivityDataset path derives this from the data (see ActivityDataset.is_home_col_idx).
_LEGACY_IS_HOME_COL_IDX = 37


type Fingerprint = dict


def is_home_node_mask(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """Boolean mask over the rows of ``x``, ``True`` at each user's own home node."""
    return (x[..., is_home_idx] > 0.0).bool()


class GenevaDataset(pyg.data.InMemoryDataset):
    """Legacy in-memory dataset: wraps a list of per-user PyG graphs loaded from a pickle file."""

    def __init__(self, dataset_path: Path):
        super().__init__()

        print("Loading dataset...")

        with open(dataset_path, "rb") as f:
            graphs = pickle.load(f)

        print(f"Loaded {len(graphs)} graphs. Applying transforms...")

        self._graphs = graphs

        print("Transforms applied.")

    @property
    def num_classes(self):
        return 1  # self._infer_num_classes(self._graphs[0].y)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int):
        return self._graphs[idx]


class ActivityDataset(pyg.data.Dataset):
    """Main PyG dataset for a travel survey: shared network graph + per-user spatial features/labels + demographics.

    Stores the network graph once and slices per-user tensors on ``get()`` call.
    Each item returned by ``get(i)`` is a ``pyg.data.Data`` with ``x = [network_features || spatial_features[i]]``,
    ``y = spatial_labels[i]``, and ``graph_x = demographics[i]``.
    """

    PROCESSED_FILE_NAMES = [
        "network_graph.pt",
        "spatial_features.pt",
        "spatial_labels.pt",
        "demographics.pt",
        "distances.pt",
        "group_ids.pt",
    ]

    def __init__(
        self,
        root: str,
        network_graph: pyg.data.Data | None = None,
        spatial_features: torch.Tensor | None = None,
        spatial_labels: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
        distances: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._network_graph_in = network_graph
        self._spatial_features_in = spatial_features
        self._spatial_labels_in = spatial_labels
        self._demographics_in = demographics
        self._distances_in = distances
        self._group_ids_in = group_ids

        super().__init__(root, transform, pre_transform, pre_filter)
        del (
            self._network_graph_in,
            self._spatial_features_in,
            self._spatial_labels_in,
            self._demographics_in,
            self._distances_in,
            self._group_ids_in,
        )

        processed_dir = Path(self.processed_dir)
        self.network_graph: pyg.data.Data = torch.load(processed_dir / "network_graph.pt", weights_only=False)
        self.spatial_features: torch.Tensor = torch.load(processed_dir / "spatial_features.pt", weights_only=True)
        self.spatial_labels: torch.Tensor = torch.load(processed_dir / "spatial_labels.pt", weights_only=True)
        self.demographics: torch.Tensor = torch.load(processed_dir / "demographics.pt", weights_only=True)
        self.distances: torch.Tensor = torch.load(processed_dir / "distances.pt", weights_only=True)
        self.group_ids: torch.Tensor = torch.load(processed_dir / "group_ids.pt", weights_only=True)

        # Check that each user has one and only one home node
        home_counts = self.spatial_features[:, :, self.is_home_spatial_idx].sum(dim=1)
        if not torch.all(home_counts == 1):
            num_bad = int((home_counts != 1).sum())
            raise ValueError(f"{num_bad} users do not have exactly one home node. Expected one per user.")

        # Check that the graph actually has nodes
        num_nodes = self.network_graph.num_nodes
        if num_nodes is None:
            raise ValueError("Network graph has `None` number of nodes")

        self.num_individuals: int = self.demographics.size(0)
        self.num_nodes: int = num_nodes
        self._is_scaled: bool = False

    @property
    def num_network_features(self) -> int:
        """Number of network node-feature columns (those preceding the spatial block in ``x``)."""
        return self.network_graph.x.shape[1]

    @property
    def is_home_spatial_idx(self) -> int:
        """Column index of ``is_home`` within the per-user spatial feature block."""
        return SPATIAL_FEATURE_NAMES.index("is_home")

    @property
    def is_home_col_idx(self) -> int:
        """Column index of ``is_home`` within the concatenated node feature matrix ``x``."""
        return self.num_network_features + self.is_home_spatial_idx

    @cached_property
    def home_hop_distance(self) -> np.ndarray:
        """All-pairs hop distance matrix (over spatial adjacency graph) ``[num_nodes, num_nodes]`` for the network graph."""
        return compute_home_hop_distance(self.network_graph.edge_index, self.num_nodes)

    @cached_property
    def median_realised_size(self) -> int:
        """Median number of visited nodes per user over the dataset, counting the home node."""
        sizes = [int(data.y.sum()) for data in self]
        return int(np.median(sizes))

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return self.PROCESSED_FILE_NAMES

    def download(self) -> None:
        pass

    def process(self):
        graph = self._network_graph_in
        sf = self._spatial_features_in
        sl = self._spatial_labels_in
        demo = self._demographics_in
        dist = self._distances_in
        group_ids = self._group_ids_in

        if graph is None or sf is None or sl is None or demo is None or dist is None or group_ids is None:
            raise ValueError("First-time construction requires all components Tensors/Data to be passed to init.")

        num_sf_users, num_sf_nodes, _ = sf.shape
        num_sl_users, num_sl_nodes, num_sl_labels = sl.shape
        num_dm_users, _ = demo.shape
        num_di_users, num_di_nodes, num_di_feat = dist.shape
        num_grp_users = group_ids.shape[0]

        if not (graph.num_nodes == num_sf_nodes == num_sl_nodes == num_di_nodes):
            raise ValueError(
                f"Node count mismatch: network has {graph.num_nodes}, spatial_features has {num_sf_nodes}, "
                f"spatial_labels has {num_sl_nodes}, distances has {num_di_nodes}."
            )
        if not (num_sf_users == num_sl_users == num_dm_users == num_di_users == num_grp_users):
            raise ValueError(
                f"Individual count mismatch: spatial_features has {num_sf_users}, spatial_labels has {num_sl_users}, "
                f"demographics has {num_dm_users}, distances has {num_di_users}."
            )
        if num_sl_labels != 1:
            raise ValueError(f"spatial_labels last dim must be 1, got {num_sl_labels}.")
        if num_di_feat != 1:
            raise ValueError(f"distances last dim must be 1, got {num_di_feat}.")

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        processed_dir = Path(self.processed_dir)
        torch.save(graph, processed_dir / "network_graph.pt")
        torch.save(sf, processed_dir / "spatial_features.pt")
        torch.save(sl, processed_dir / "spatial_labels.pt")
        torch.save(demo, processed_dir / "demographics.pt")
        torch.save(dist, processed_dir / "distances.pt")
        torch.save(group_ids, processed_dir / "group_ids.pt")

    def len(self) -> int:
        return self.num_individuals

    def get(self, i: int) -> pyg.data.Data:
        if not 0 <= i < self.num_individuals:
            raise IndexError(f"Index {i} out of range for {self.num_individuals} individuals.")

        network_x = self.network_graph.x
        spatial_x = self.spatial_features[i]

        # Cast spatial features to same type as network graph features if mismatch
        if spatial_x.dtype != network_x.dtype:
            spatial_x = spatial_x.to(network_x.dtype)

        # Create new individual-annotated network graph
        full_x = torch.cat([network_x, spatial_x], dim=1)
        y = self.spatial_labels[i]
        graph_x = self.demographics[i]
        edge_index = self.network_graph.edge_index
        edge_attr = self.network_graph.edge_attr

        data = pyg.data.Data(x=full_x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Unsqueeze so batching works
        data.graph_x = graph_x.unsqueeze(0)
        data.user_id = torch.tensor([i], dtype=torch.long)

        # Per-user distance-from-home
        data.distances = self.distances[i].to(full_x.dtype)

        # Carry over any extra attributes from the network graph (pos, etc.)
        for key, value in self.network_graph:
            if key in ("x", "edge_index", "edge_attr", "y"):
                continue

            data[key] = value

        return data


@dataclass
class FittedScalers:
    """Container for ``StandardScaler`` instances fitted on the training split."""

    network_features: StandardScaler
    network_edges: StandardScaler
    spatial: StandardScaler | None
    demographics: StandardScaler
    distances: StandardScaler
    exclude_spatial_cols: list[int]
    fingerprint: Fingerprint

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FittedScalers":
        with path.open("rb") as f:
            return pickle.load(f)


def load_dataset(
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

    dataset = load_or_build_dataset(
        cfg, project_root=project_root, pre_transform=positional_encodings_transforms, **build_kwargs
    )

    fingerprint = dataset_fingerprint(cfg, dataset)

    train_idx, val_idx, test_idx = split_indices(
        dataset, val_size, test_size, seed, cache_path=splits_cache, fingerprint=fingerprint
    )

    scalers = None
    if scalers_cache.exists():
        scalers = FittedScalers.load(scalers_cache)

    if scalers is None or scalers.fingerprint != fingerprint:
        scalers = fit_scalers(
            dataset, train_idx, exclude_spatial_cols=[dataset.is_home_spatial_idx], fingerprint=fingerprint
        )
        scalers.save(scalers_cache)

    apply_scalers(dataset, scalers)

    return (
        cast(ActivityDataset, dataset[train_idx]),
        cast(ActivityDataset, dataset[val_idx]),
        cast(ActivityDataset, dataset[test_idx]),
        scalers,
    )


def load_or_build_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **build_kwargs,
) -> ActivityDataset:
    """Construct an ``ActivityDataset`` rooted at ``cfg.data.paths.pyg_datasets``."""
    project_root = get_project_root(project_root)
    dataset_path = project_root / cfg.data.paths.pyg_datasets

    if not all((dataset_path / file).exists() for file in ActivityDataset.PROCESSED_FILE_NAMES):
        data, network_nodes, network_edges = load_data(cfg.data, project_root)
        network_graph, spatial_features, spatial_labels, demographics, distances, group_ids = convert_to_torch(
            data, network_nodes, network_edges
        )
        dataset_path = project_root / cfg.data.paths.pyg_datasets

        return ActivityDataset(
            str(dataset_path),
            network_graph,
            spatial_features,
            spatial_labels,
            demographics,
            distances,
            group_ids,
            **build_kwargs,
        )

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
    exclude_spatial_cols: list[int],
    exclude_demographic_cols: list[int] | None = None,
) -> FittedScalers:
    """Fit ``StandardScaler`` instances on the training split of each feature group.

    Args:
        dataset: The full ``ActivityDataset`` (not yet scaled).
        train_idx: Indices of training individuals.
        exclude_spatial_cols: Column indices excluded from spatial scaler fitting (e.g. ``[is_home_spatial_idx]``).
        exclude_demographic_cols: Column indices excluded from demographics scaler fitting.
        fingerprint: Identity of the dataset being fitted on, stored for cache validation.

    Returns:
        ``FittedScalers`` with one scaler per feature group (network nodes, edges, spatial, demographics).
    """
    # Network node features: no need to separate between train and test, the network is the same.
    network_scaler = StandardScaler()
    network_scaler.fit(dataset.network_graph.x.numpy())

    # Network edge attributes: same as above
    edge_scaler = StandardScaler()
    edge_scaler.fit(dataset.network_graph.edge_attr.numpy())

    # Spatial features: individual-specific, fit only on train. Skip when every column is excluded (e.g. only is_home).
    spatial_train = dataset.spatial_features[train_idx]
    keep = [c for c in range(spatial_train.shape[-1]) if c not in exclude_spatial_cols]
    spatial_scaler = None
    if keep:
        spatial_flat = spatial_train[..., keep].reshape(-1, len(keep)).float().numpy()
        spatial_scaler = StandardScaler()
        spatial_scaler.fit(spatial_flat)

    # Demographics: individual-specific, fit only on train
    demo_train = dataset.demographics[train_idx]
    if exclude_demographic_cols:
        keep = [c for c in range(demo_train.shape[-1]) if c not in exclude_demographic_cols]
        demo_train = demo_train[..., keep]
    demo_scaler = StandardScaler()
    demo_scaler.fit(demo_train.float().numpy())

    # Distance-from-home: individual-specific, fit only on train
    distances_train = dataset.distances[train_idx]
    distances_scaler = StandardScaler()
    distances_scaler.fit(distances_train.reshape(-1, distances_train.shape[-1]).float().numpy())

    return FittedScalers(
        network_features=network_scaler,
        network_edges=edge_scaler,
        spatial=spatial_scaler,
        demographics=demo_scaler,
        distances=distances_scaler,
        exclude_spatial_cols=exclude_spatial_cols,
        fingerprint=fingerprint,
    )


def apply_scalers(dataset: ActivityDataset, scalers: FittedScalers) -> None:
    """Apply fitted scalers to the dataset in-place, transforming network/spatial/demographic tensors."""
    if dataset._is_scaled:
        raise RuntimeError("apply_scalers called twice on the same ActivityDataset")

    x = dataset.network_graph.x.numpy()
    dataset.network_graph.x = torch.from_numpy(scalers.network_features.transform(x)).float()

    ea = dataset.network_graph.edge_attr.numpy()
    dataset.network_graph.edge_attr = torch.from_numpy(scalers.network_edges.transform(ea)).float()

    if scalers.spatial is not None:
        sf = dataset.spatial_features.float()
        n_cols = sf.shape[-1]
        keep = [c for c in range(n_cols) if c not in scalers.exclude_spatial_cols]
        flat = sf.reshape(-1, n_cols)[:, keep].numpy()
        scaled_keep = torch.from_numpy(scalers.spatial.transform(flat)).float()
        result = sf.reshape(-1, n_cols).clone()
        result[:, keep] = scaled_keep
        dataset.spatial_features = result.reshape(sf.shape)

    demo = dataset.demographics.float().numpy()
    dataset.demographics = torch.from_numpy(scalers.demographics.transform(demo)).float()

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
        "n_spatial_cols": int(dataset.spatial_features.shape[-1]),
        "n_demographic_cols": int(dataset.demographics.shape[-1]),
        "n_groups": len(dataset.group_ids.unique()),
        "data_config": hashlib.blake2b(config_str.encode(), digest_size=16).hexdigest(),
    }


def load_gva_dataset(
    cfg: Config, test_size: float, seed: int, project_root: Path | None = None, graphs_name: str = "Graphs"
) -> tuple[GenevaDataset, GenevaDataset]:
    project_root: Path = get_project_root(project_root)

    pyg_path = project_root / cfg.data.paths.pyg_datasets
    graphs_path = pyg_path / f"{graphs_name}.pickle"

    train_path = pyg_path / "train.pickle"
    test_path = pyg_path / "test.pickle"

    if train_path.exists() and test_path.exists():
        print(f"Loading {pyg_path} dataset...")

        with open(train_path, "rb") as f:
            train_dataset: GenevaDataset = pickle.load(f)

        with open(test_path, "rb") as f:
            test_dataset: GenevaDataset = pickle.load(f)

        return train_dataset, test_dataset

    print(f"Loading {graphs_path} graph list and building dataset...")

    dataset = GenevaDataset(graphs_path)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)

    # noinspection PyTypeChecker
    train_dataset: GenevaDataset = dataset[train_indices]
    # noinspection PyTypeChecker
    test_dataset: GenevaDataset = dataset[test_indices]

    print(f"Train size: {len(train_dataset)}. Splitting dataset: ")

    num_features = train_dataset[0].x.shape[1]
    non_home_cols = [i for i in range(num_features) if i != _LEGACY_IS_HOME_COL_IDX]

    train_x = torch.cat([g.x[:, non_home_cols] for g in train_dataset]).numpy()
    train_edge_attr = torch.cat([g.edge_attr for g in train_dataset]).numpy()

    x_scaler = StandardScaler()
    x_scaler.fit(train_x)

    e_scaler = StandardScaler()
    e_scaler.fit(train_edge_attr)

    print("Fitted scalers. Processing dataset:")

    # Transform both splits
    for g in tqdm(train_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    for g in tqdm(test_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    print("Writing processed datasets")

    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)

    print("Done.")

    return train_dataset, test_dataset


def replace_scaled_features(g: BaseData, e_scaler: StandardScaler, x_scaler: StandardScaler, non_home_cols: list[int]):
    """Scale ``g.x`` (excluding ``is_home``) and ``g.edge_attr`` in-place using the provided scalers."""
    g.x[:, non_home_cols] = torch.tensor(x_scaler.transform(g.x[:, non_home_cols].numpy()), dtype=torch.float)
    g.edge_attr = torch.tensor(e_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
