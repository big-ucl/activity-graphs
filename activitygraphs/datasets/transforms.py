"""FeatureTransforms: the per-column transforms of the feature tensors, fitted on the training split."""

import logging
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from activitygraphs.schema import ColumnKind, DatasetSchema, TensorSchema
from activitygraphs.tensors import Fingerprint, LocationTensors, UserTensors

log = logging.getLogger(__name__)


class Transform(StrEnum):
    """Transform applied to a feature column."""

    PASSTHROUGH = "passthrough"
    ZSCORE = "zscore"
    LOG1P_ZSCORE = "log1p_zscore"


@dataclass(frozen=True)
class TransformPolicy:
    """Defines the transform for each column kind, and records the per-group overrides.

    Attributes:
        kinds: Transform of the columns of each kind.
        group_overrides: dict of overridden column transforms (i.e. {group -> (column kind -> new transform)}).
    """

    kinds: Mapping[ColumnKind, Transform]
    group_overrides: Mapping[str, Mapping[ColumnKind, Transform]]

    def column_transforms(self, schema: TensorSchema) -> tuple[Transform, ...]:
        """Return the transform of each column of ``schema`` in column order."""
        overrides = self.group_overrides.get(schema.group, {})
        group_kinds = {**self.kinds, **overrides}

        return tuple(group_kinds[column.kind] for column in schema.columns)


TRANSFORM_POLICY = TransformPolicy(
    kinds={
        ColumnKind.CONTINUOUS: Transform.ZSCORE,
        ColumnKind.COUNT: Transform.LOG1P_ZSCORE,
        ColumnKind.BINARY: Transform.PASSTHROUGH,
        ColumnKind.CATEGORICAL: Transform.PASSTHROUGH,
    },
    group_overrides={},
)


@dataclass
class GroupTransform:
    """Defines the column transforms for one tensor group, and stores the fitted scaler if applicable.

    Attributes:
        transforms: Transform of each column, in column order.
        scaler: ``StandardScaler`` fitted over the columns that are not passed through, in column order.
        ``None`` when every column is passed through.
    """

    transforms: tuple[Transform, ...]
    scaler: StandardScaler | None

    @classmethod
    def fit(cls, schema: TensorSchema, transforms: tuple[Transform, ...], values: torch.Tensor) -> Self:
        """Fit the scaler for a group on ``values``.

        Warns about every constant z-scored column.

        Args:
            schema: Schema of the group.
            transforms: Transform of each column of the group.
            values: tensor of values to fit the scaler, shape ``[..., num_columns]`` .
        """
        scaled_idx = _scaled_column_indices(transforms)
        if not scaled_idx:
            return cls(transforms, None)

        flat = _log1p(values, transforms)[:, scaled_idx]
        scaler = StandardScaler().fit(flat.numpy())
        variances = cast(np.ndarray, scaler.var_)

        for column, variance in zip(scaled_idx, variances, strict=True):
            if variance == 0:
                name = schema.columns[column].name
                log.warning("Column `%s.%s` is constant on the fitted rows and is z-scored to 0.", schema.group, name)

        return cls(transforms, scaler)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` transformed column by column, as float32, in the shape of ``values``."""
        flat = _log1p(values, self.transforms)

        if self.scaler is not None:
            scaled_idx = _scaled_column_indices(self.transforms)
            flat[:, scaled_idx] = torch.from_numpy(self.scaler.transform(flat[:, scaled_idx].numpy())).float()

        return flat.reshape(values.shape)


@dataclass
class FeatureTransforms:
    """The transforms of every feature group, fitted on the training split.

    Attributes:
        groups: The transforms of each feature group, keyed by group.
        policy: The ``TRANSFORM_POLICY`` used to fit the transforms.
        fingerprint: Identity of the dataset that was fit on, for cache validation.
    """

    groups: dict[str, GroupTransform]
    policy: TransformPolicy
    fingerprint: Fingerprint

    def is_current(self, fingerprint: Fingerprint) -> bool:
        """Return whether the transforms were fitted on the dataset ``fingerprint`` under the current ``TRANSFORM_POLICY``."""
        return self.fingerprint == fingerprint and self.policy == TRANSFORM_POLICY

    def save(self, path: Path) -> None:
        """Pickle the transforms to ``path``."""
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def read(path: Path) -> "FeatureTransforms":
        """Read transforms pickled by ``save``."""
        with path.open("rb") as f:
            return pickle.load(f)


def fit_transforms(
    locations: LocationTensors,
    users: UserTensors,
    schema: DatasetSchema,
    train_idx: list[int],
    fingerprint: Fingerprint,
) -> FeatureTransforms:
    """Fit the transforms of each feature group of ``locations`` and ``users``, under the current ``TRANSFORM_POLICY``.

    Args:
        locations: The location tensors.
        users: The user tensors.
        schema: The column schema of each tensor group.
        train_idx: Indices of training users.
        fingerprint: Identity of the dataset being fitted on, stored for cache validation.

    Returns:
        ``FeatureTransforms`` with one ``GroupTransform`` per feature group.

    Raises:
        ValueError: The policy transforms a ``user_nodes`` column.
    """
    policy = TRANSFORM_POLICY

    user_node_transforms = policy.column_transforms(schema.user_nodes.numeric)
    if _scaled_column_indices(user_node_transforms):
        raise ValueError("The `user_nodes` columns are built from the home node and cannot be transformed.")

    train_homes = users.home_node[train_idx]

    # Location and edge features: no need to separate between train and test, the graph is the same.
    values = {
        "locations": locations.x,
        "edges": locations.edge_attr,
        "user_attributes": users.attributes[train_idx],
        "distances": locations.distance_m[train_homes].unsqueeze(-1),
    }

    groups = {"user_nodes": GroupTransform(user_node_transforms, None)}
    for group_schema in schema.feature_groups:
        if group_schema is schema.user_nodes:
            continue

        numeric = group_schema.numeric
        transforms = policy.column_transforms(numeric)
        groups[group_schema.group] = GroupTransform.fit(numeric, transforms, values[group_schema.group])

    return FeatureTransforms(groups=groups, policy=policy, fingerprint=fingerprint)


def _scaled_column_indices(transforms: tuple[Transform, ...]) -> list[int]:
    """Return the indices of the columns that are z-scored."""
    return [idx for idx, transform in enumerate(transforms) if transform != Transform.PASSTHROUGH]


def _log1p(values: torch.Tensor, transforms: tuple[Transform, ...]) -> torch.Tensor:
    """Return a float32 copy of ``values`` flattened to ``[-1, num_columns]``, with log1p applied to its log columns."""
    flat = values.float().flatten(end_dim=-2).clone()
    log_columns = [idx for idx, transform in enumerate(transforms) if transform == Transform.LOG1P_ZSCORE]
    flat[:, log_columns] = torch.log1p(flat[:, log_columns])

    return flat
