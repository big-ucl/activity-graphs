"""FittedScalers: the feature scalers fitted on the training split, and their fit and in-place application."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from activitygraphs.datasets.dataset import Fingerprint, VisitDataset


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


def fit_scalers(
    dataset: VisitDataset,
    train_idx: list[int],
    fingerprint: Fingerprint,
    exclude_user_node_cols: list[int],
) -> FittedScalers:
    """Fit ``StandardScaler`` instances on the training split of each feature group.

    Args:
        dataset: The full ``VisitDataset`` (not yet scaled).
        train_idx: Indices of training users.
        exclude_user_node_cols: Column indices excluded from user node scaler fitting (e.g. ``[is_home_user_node_idx]``).
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


def apply_scalers(dataset: VisitDataset, scalers: FittedScalers) -> None:
    """Apply fitted scalers to the dataset in-place, transforming graph/user node/user attribute tensors."""
    if dataset._is_scaled:
        raise RuntimeError("apply_scalers called twice on the same VisitDataset")

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
