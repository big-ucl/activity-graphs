"""Typed column metadata for the dataset tensors: specs for columns, tensors, and dataset."""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

log = logging.getLogger(__name__)


class ColumnKind(StrEnum):
    """Specifies the kind of data in a tensor column, for downstream processing/scaling."""

    CONTINUOUS = "continuous"
    COUNT = "count"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class ColumnSpec:
    """Specifies a tensor column: name, kind and, for a categorical column, the categories.

    Attributes:
        name: Column name in the source frame.
        kind: Kind of data held by the column.
        categories: Categories (string values) of a categorical column. ``categories[i]`` is encoded as ``i + 1``,
          missing or unknown values are encoded as 0. Empty for other kinds of column.

    Raises:
        ValueError: A categorical column has no or repeated categories, or a column of another kind has categories.
    """

    name: str
    kind: ColumnKind
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the categories against the kind."""
        if self.kind == ColumnKind.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"Categorical column `{self.name}` has no categories.")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError(f"Categorical column `{self.name}` repeats a category: {self.categories}.")
        elif self.categories:
            raise ValueError(f"Column `{self.name}` of kind {self.kind} cannot have categories.")

    @property
    def cardinality(self) -> int:
        """Number of codes of a categorical column, including the missing-or-unknown code 0.

        Raises:
            ValueError: The column is not categorical.
        """
        if self.kind != ColumnKind.CATEGORICAL:
            raise ValueError(f"Column `{self.name}` of kind {self.kind} has no cardinality.")

        return len(self.categories) + 1

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-serialisable dict."""
        return {"name": self.name, "kind": str(self.kind), "categories": list(self.categories)}

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> Self:
        """Return the spec written by ``to_dict``."""
        return cls(spec["name"], ColumnKind(spec["kind"]), tuple(spec["categories"]))


@dataclass(frozen=True)
class TensorSchema:
    """Column specs of a tensor group (locations, user_attributes, ...), in order of tensor columns.

    Attributes:
        group: Name of the tensor group, i.e. one of the fields of ``DatasetSchema``.
        columns: list of specs, for each tensor column, in column order.

    Raises:
        ValueError: Two columns share a name.
    """

    group: str
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        """Check that the column names are unique."""
        column_names = [column.name for column in self.columns]
        repeated_names = sorted({name for name in column_names if column_names.count(name) > 1})
        if repeated_names:
            raise ValueError(f"Group `{self.group}` repeats columns {repeated_names}.")

    @property
    def names(self) -> list[str]:
        """Column names, in column order."""
        return [column.name for column in self.columns]

    @property
    def width(self) -> int:
        """Schema width, i.e. number of columns."""
        return len(self.columns)

    def index_of(self, name: str) -> int:
        """Return the column index of ``name``.

        Raises:
            KeyError: The group has no column ``name``.
        """
        for idx, column in enumerate(self.columns):
            if column.name == name:
                return idx

        raise KeyError(f"Group `{self.group}` has no column `{name}`; its columns are {self.names}.")

    def indices(self, *kinds: ColumnKind) -> list[int]:
        """Return the indices of the columns of any of ``kinds``, in column order."""
        return [idx for idx, column in enumerate(self.columns) if column.kind in kinds]

    def to_dict(self) -> dict[str, Any]:
        """Return the schema as a JSON-serialisable dict."""
        return {"group": self.group, "columns": [column.to_dict() for column in self.columns]}

    @classmethod
    def from_dict(cls, schema: dict[str, Any]) -> Self:
        """Return the schema written by ``to_dict``."""
        return cls(schema["group"], tuple(ColumnSpec.from_dict(column) for column in schema["columns"]))


@dataclass(frozen=True)
class DatasetSchema:
    """Schema of the tensors of a ``VisitDataset``, one schema per tensor group.

    Attributes:
        locations: Location node features.
        edges: Edge features.
        user_nodes: Per-user node features.
        user_attributes: Per-user attributes.
        distances: Per-user distance from home of every node.
        targets: Per-user visit labels.

    Raises:
        ValueError: A schema's ``group`` is not the field it is stored in.
    """

    locations: TensorSchema
    edges: TensorSchema
    user_nodes: TensorSchema
    user_attributes: TensorSchema
    distances: TensorSchema
    targets: TensorSchema

    def __post_init__(self) -> None:
        """Check that each schema is labelled with the group it is stored in."""
        for field, schema in zip(fields(self), self.groups, strict=True):
            if schema.group != field.name:
                raise ValueError(f"The `{field.name}` schema is labelled with group `{schema.group}`.")

    @property
    def groups(self) -> tuple[TensorSchema, ...]:
        """Return the schema of every tensor group, in field order."""
        return tuple(getattr(self, field.name) for field in fields(self))

    def to_dict(self) -> dict[str, Any]:
        """Return the schemas as a JSON-serialisable dict, keyed by group."""
        return {schema.group: schema.to_dict() for schema in self.groups}

    def save(self, path: Path) -> None:
        """Write the schemas to ``path`` as JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read the schemas written by ``save``."""
        schemas = json.loads(path.read_text())

        return cls(**{field.name: TensorSchema.from_dict(schemas[field.name]) for field in fields(cls)})


def infer_schema(
    group: str, frame: pd.DataFrame, columns: Sequence[str], overrides: Mapping[str, ColumnKind]
) -> TensorSchema:
    """Return the schema of ``columns`` of ``frame``, and infers the kind of each column not in ``overrides``.

    Column kinds are inferred as follows:
          - binary if it is boolean or numeric with values in {0, 1},
          - categorical if it holds strings or a pandas categorical,
          - continuous otherwise.
    The categories of a categorical column are the distinct, non-missing values, as sorted strings.

    Each inferred kind is logged as a warning.

    Args:
        group: Name of the tensor group.
        frame: Frame holding the columns.
        columns: Columns of the tensor, in tensor column order.
        overrides: Column kinds that will be overridden.

    Returns:
        The schema of ``columns``.

    Raises:
        KeyError: An override names a column outside ``columns``.
        ValueError: A binary column holds values other than 0 and 1, or a count column holds negative values.
    """
    unknown_cols = sorted(set(overrides) - set(columns))
    if unknown_cols:
        raise KeyError(
            f"Overrides for group `{group}` name columns {unknown_cols}, which are not among {list(columns)}."
        )

    specs = []
    for name in columns:
        series = frame[name]
        if name in overrides:
            kind = overrides[name]
        else:
            kind = _infer_kind(series)
            log.warning("Inferred `%s.%s` (dtype %s) as %s.", group, name, series.dtype, kind)
        specs.append(_column_spec(name, kind, series))

    return TensorSchema(group, tuple(specs))


def _infer_kind(series: pd.Series) -> ColumnKind:
    """Return the kind of ``series`` inferred from the column dtype and values."""
    if is_bool_dtype(series):
        return ColumnKind.BINARY
    if is_numeric_dtype(series):
        values = series.dropna()
        if len(values) > 0 and values.isin([0, 1]).all():
            return ColumnKind.BINARY
        return ColumnKind.CONTINUOUS

    return ColumnKind.CATEGORICAL


def _column_spec(name: str, kind: ColumnKind, series: pd.Series) -> ColumnSpec:
    """Return the spec of column ``name`` of kind ``kind``, checking its values against the kind."""
    values = series.dropna()
    if kind == ColumnKind.BINARY and not values.isin([0, 1]).all():
        raise ValueError(f"Binary column `{name}` holds values other than 0 and 1.")
    if kind == ColumnKind.COUNT and (values < 0).any():
        raise ValueError(f"Count column `{name}` holds negative values.")
    if kind == ColumnKind.CATEGORICAL:
        return ColumnSpec(name, kind, tuple(sorted(values.astype(str).unique())))

    return ColumnSpec(name, kind)
