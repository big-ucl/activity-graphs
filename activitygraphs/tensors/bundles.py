"""LocationTensors and UserTensors: the raw tensors of a survey, one bundle for the locations and one for the users."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import polars as pl
import torch

from activitygraphs.schema import DatasetSchema


class LocationFile(StrEnum):
    """Names of the files of a ``LocationTensors`` bundle, in its cache directory."""

    X = "x.pt"
    ACCESS = "access.pt"
    EDGE_INDEX = "edge_index.pt"
    EDGE_ATTR = "edge_attr.pt"
    DISTANCE_M = "distance_m.pt"
    NODE_IDS = "node_ids.parquet"


class UserFile(StrEnum):
    """Names of the files of a ``UserTensors`` bundle, in its cache directory."""

    HOME_NODE = "home_node.pt"
    VISITED = "visited.pt"
    ATTRIBUTES = "attributes.pt"
    ATTRIBUTES_CAT = "attributes_cat.pt"
    GROUP_IDS = "group_ids.pt"
    USER_IDS = "user_ids.parquet"


@dataclass(frozen=True)
class LocationTensors:
    """The raw tensors of the shared location graph of a survey.

    Attributes:
        x: Location features, ``[num_nodes, num_location_features]``.
        access: Accessibility features of the locations, ``[num_nodes, num_access_features]``.
        edge_index: Edge list, ``[2, num_edges]``.
        edge_attr: Edge features, ``[num_edges, num_edge_features]``.
        distance_m: Distance between the centroids of every pair of nodes in metres, ``[num_nodes, num_nodes]``.
            Row ``i`` holds the distances from node ``i`` to every node.
        node_ids: Survey ``loc_id`` of each node, one row per node.
    """

    x: torch.Tensor
    access: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    distance_m: torch.Tensor
    node_ids: pl.DataFrame

    @property
    def num_nodes(self) -> int:
        """Number of nodes of the graph."""
        return self.node_ids.height

    def save(self, directory: Path) -> None:
        """Write the tensors to ``directory``, one file each."""
        directory.mkdir(parents=True, exist_ok=True)
        tensor_files = {
            LocationFile.X: self.x,
            LocationFile.ACCESS: self.access,
            LocationFile.EDGE_INDEX: self.edge_index,
            LocationFile.EDGE_ATTR: self.edge_attr,
            LocationFile.DISTANCE_M: self.distance_m,
        }
        for file, tensor in tensor_files.items():
            torch.save(tensor, directory / file)

        self.node_ids.write_parquet(directory / LocationFile.NODE_IDS)

    @classmethod
    def read(cls, directory: Path) -> Self:
        """Read the tensors written by ``save`` to ``directory``."""
        return cls(
            x=torch.load(directory / LocationFile.X, weights_only=True),
            access=torch.load(directory / LocationFile.ACCESS, weights_only=True),
            edge_index=torch.load(directory / LocationFile.EDGE_INDEX, weights_only=True),
            edge_attr=torch.load(directory / LocationFile.EDGE_ATTR, weights_only=True),
            distance_m=torch.load(directory / LocationFile.DISTANCE_M, weights_only=True),
            node_ids=pl.read_parquet(directory / LocationFile.NODE_IDS),
        )


@dataclass(frozen=True)
class UserTensors:
    """The raw per-user tensors of a survey, one row per user.

    Attributes:
        home_node: Home node of each user (as node indices), int64, ``[num_users]``.
        visited: Visit labels of each user over every node, ``[num_users, num_nodes]``.
        attributes: Numeric user attributes, ``[num_users, n_numeric_attr]``.
        attributes_cat: Categorical user attribute codes, int64, ``[num_users, n_categorical_attr]``.
        group_ids: Household group id of each user, ``[num_users]``.
        user_ids: Survey ``user_id`` and ``hh_id`` of each user, one row per user.
    """

    home_node: torch.Tensor
    visited: torch.Tensor
    attributes: torch.Tensor
    attributes_cat: torch.Tensor
    group_ids: torch.Tensor
    user_ids: pl.DataFrame

    @property
    def num_users(self) -> int:
        """Number of users."""
        return self.user_ids.height

    def save(self, directory: Path) -> None:
        """Write the tensors to ``directory``, one file each."""
        directory.mkdir(parents=True, exist_ok=True)
        tensor_files = {
            UserFile.HOME_NODE: self.home_node,
            UserFile.VISITED: self.visited,
            UserFile.ATTRIBUTES: self.attributes,
            UserFile.ATTRIBUTES_CAT: self.attributes_cat,
            UserFile.GROUP_IDS: self.group_ids,
        }
        for file, tensor in tensor_files.items():
            torch.save(tensor, directory / file)

        self.user_ids.write_parquet(directory / UserFile.USER_IDS)

    @classmethod
    def read(cls, directory: Path) -> Self:
        """Read the tensors written by ``save`` to ``directory``."""
        return cls(
            home_node=torch.load(directory / UserFile.HOME_NODE, weights_only=True),
            visited=torch.load(directory / UserFile.VISITED, weights_only=True),
            attributes=torch.load(directory / UserFile.ATTRIBUTES, weights_only=True),
            attributes_cat=torch.load(directory / UserFile.ATTRIBUTES_CAT, weights_only=True),
            group_ids=torch.load(directory / UserFile.GROUP_IDS, weights_only=True),
            user_ids=pl.read_parquet(directory / UserFile.USER_IDS),
        )


type SurveyTensors = tuple[LocationTensors, UserTensors, DatasetSchema]


def check_tensors(locations: LocationTensors, users: UserTensors, schema: DatasetSchema) -> None:
    """Check that the bundles agree on the number of users and nodes, and that their widths match ``schema``.

    Raises:
        ValueError: The tensors and ids disagree on the number of users or nodes, a group's tensors have a different
            number of numeric or categorical columns than its schema, a home node is not a node of the graph, or a
            feature tensor holds a NaN or infinite value.
    """
    _require_equal_counts(
        "User",
        {
            "home_node": users.home_node.shape[0],
            "visited": users.visited.shape[0],
            "attributes": users.attributes.shape[0],
            "attributes_cat": users.attributes_cat.shape[0],
            "group_ids": users.group_ids.shape[0],
            "user_ids": users.user_ids.height,
        },
    )
    _require_equal_counts(
        "Node",
        {
            "x": locations.x.shape[0],
            "access": locations.access.shape[0],
            "visited": users.visited.shape[1],
            "distance_m (rows)": locations.distance_m.shape[0],
            "distance_m (columns)": locations.distance_m.shape[1],
            "node_ids": locations.node_ids.height,
        },
    )

    numeric_and_categorical_widths = {
        "locations": (locations.x.shape[-1], 0),
        "access": (locations.access.shape[-1], 0),
        "edges": (locations.edge_attr.shape[-1], 0),
        "user_attributes": (users.attributes.shape[-1], users.attributes_cat.shape[-1]),
        "distances": (1, 0),
        "targets": (1, 0),
    }

    for group_schema in schema.groups:
        if group_schema is schema.user_nodes:
            continue

        widths = numeric_and_categorical_widths[group_schema.group]
        schema_widths = (group_schema.numeric.width, group_schema.categorical.width)
        if widths != schema_widths:
            raise ValueError(
                f"The `{group_schema.group}` tensors have (numeric, categorical) widths {widths} but its schema has "
                f"{schema_widths}."
            )

    num_outside = int(((users.home_node < 0) | (users.home_node >= locations.num_nodes)).sum())
    if num_outside:
        raise ValueError(f"{num_outside} users have a home node outside the graph.")

    _require_finite({
        "x": locations.x,
        "access": locations.access,
        "edge_attr": locations.edge_attr,
        "distance_m": locations.distance_m,
        "attributes": users.attributes,
    })


def _require_equal_counts(count_name: str, counts: dict[str, int]) -> None:
    """Raise unless every count in ``counts`` is the same.

    Args:
        count_name: The display name of the count that is being checked (e.g. "User").
        counts: The count read from each tensor or id frame, keyed by the tensor/frame name.

    Raises:
        ValueError: The counts differ.
    """
    if len(set(counts.values())) > 1:
        listed = ", ".join(f"{name} {count}" for name, count in counts.items())
        raise ValueError(f"{count_name} count mismatch: {listed}.")


def _require_finite(tensors: dict[str, torch.Tensor]) -> None:
    """Raise unless every value of every tensor in ``tensors`` is finite.

    Args:
        tensors: The float tensors to check, keyed by the tensor name.

    Raises:
        ValueError: A tensor holds a NaN or infinite value.
    """
    for name, tensor in tensors.items():
        num_not_finite = int((~torch.isfinite(tensor)).sum())
        if num_not_finite:
            raise ValueError(f"The `{name}` tensor holds {num_not_finite} NaN or infinite values.")
