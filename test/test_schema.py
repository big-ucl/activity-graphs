"""Unit tests for the column schema types and their JSON round-trip."""

import logging
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from activitygraphs.schema import ColumnKind, ColumnSpec, DatasetSchema, TensorSchema, infer_schema

CONTINUOUS = ColumnKind.CONTINUOUS
COUNT = ColumnKind.COUNT
BINARY = ColumnKind.BINARY
CATEGORICAL = ColumnKind.CATEGORICAL


def user_attributes() -> TensorSchema:
    return TensorSchema(
        "user_attributes",
        (
            ColumnSpec("age", CONTINUOUS),
            ColumnSpec("is_employed", BINARY),
            ColumnSpec("income_band", CATEGORICAL, ("low", "mid", "high")),
            ColumnSpec("hh_num_vehicles", CONTINUOUS),
            ColumnSpec("can_drive", BINARY),
        ),
    )


def dataset_schema() -> DatasetSchema:
    return DatasetSchema(
        locations=TensorSchema("locations", (ColumnSpec("pop_density", CONTINUOUS),)),
        edges=TensorSchema("edges", (ColumnSpec("weight", CONTINUOUS),)),
        user_nodes=TensorSchema("user_nodes", (ColumnSpec("is_home", BINARY),)),
        user_attributes=user_attributes(),
        distances=TensorSchema("distances", (ColumnSpec("distance", COUNT),)),
        targets=TensorSchema("targets", (ColumnSpec("is_visited", BINARY),)),
    )


class TestColumnSpec:
    def test_cardinality_counts_the_missing_code(self):
        assert ColumnSpec("band", CATEGORICAL, ("a", "b", "c")).cardinality == 4

    @pytest.mark.parametrize("kind", [CONTINUOUS, COUNT, BINARY])
    def test_only_categorical_columns_have_a_cardinality(self, kind):
        with pytest.raises(ValueError, match="no cardinality"):
            _ = ColumnSpec("x", kind).cardinality

    def test_a_categorical_column_needs_categories(self):
        with pytest.raises(ValueError, match="no categories"):
            ColumnSpec("band", CATEGORICAL)

    def test_categories_are_unique(self):
        with pytest.raises(ValueError, match="repeats a category"):
            ColumnSpec("band", CATEGORICAL, ("a", "b", "a"))

    @pytest.mark.parametrize("kind", [CONTINUOUS, COUNT, BINARY])
    def test_only_categorical_columns_take_categories(self, kind):
        with pytest.raises(ValueError, match="cannot have categories"):
            ColumnSpec("x", kind, ("a",))


class TestTensorSchema:
    def test_indices_select_the_columns_of_the_given_kinds_in_order(self):
        schema = user_attributes()

        assert schema.indices(BINARY) == [1, 4]
        assert schema.indices(CONTINUOUS, COUNT) == [0, 3]
        assert schema.indices(CATEGORICAL) == [2]
        assert schema.indices(COUNT) == []

    def test_index_of_finds_a_column_by_name(self):
        assert user_attributes().index_of("hh_num_vehicles") == 3

    def test_index_of_an_unknown_column_raises(self):
        with pytest.raises(KeyError, match="no column `gender`"):
            user_attributes().index_of("gender")

    def test_width_counts_the_columns(self):
        assert user_attributes().width == 5
        assert TensorSchema("targets", ()).width == 0

    def test_names_follow_column_order(self):
        schema = user_attributes()

        assert schema.names == ["age", "is_employed", "income_band", "hh_num_vehicles", "can_drive"]

    def test_column_names_are_unique(self):
        with pytest.raises(ValueError, match=r"repeats columns \['age'\]"):
            TensorSchema("user_attributes", (ColumnSpec("age", CONTINUOUS), ColumnSpec("age", COUNT)))

    def test_an_empty_group_is_allowed(self):
        schema = TensorSchema("user_attributes", ())

        assert schema.names == []
        assert schema.indices(CONTINUOUS) == []


class TestDatasetSchema:
    def test_round_trips_through_json(self, tmp_path):
        schema = dataset_schema()
        path = tmp_path / "schema.json"

        schema.save(path)

        assert DatasetSchema.read(path) == schema

    def test_round_trip_keeps_the_categories_in_code_order(self, tmp_path):
        path = tmp_path / "schema.json"
        dataset_schema().save(path)

        income = DatasetSchema.read(path).user_attributes.columns[2]

        assert income.categories == ("low", "mid", "high")
        assert income.kind is CATEGORICAL

    def test_groups_hold_one_schema_per_field_in_field_order(self):
        groups = dataset_schema().groups

        assert [schema.group for schema in groups] == [field.name for field in fields(DatasetSchema)]

    def test_a_schema_must_sit_in_its_own_group(self):
        groups = {schema.group: schema for schema in dataset_schema().groups}
        groups["edges"] = TensorSchema("locations", ())

        with pytest.raises(ValueError, match="`edges` schema is labelled with group `locations`"):
            DatasetSchema(**groups)


def location_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "is_coastal": [True, False, True, False],
        "has_station": [0, 1, 1, 0],
        "has_school": [0.0, 1.0, np.nan, 1.0],
        "pop_density": [1e-4, 3e-3, 5e-4, 2e-5],
        "num_stops": [0, 3, 12, 1],
        "zone_class": ["urban", "rural", "urban", None],
        "land_use_class": pd.Categorical(["res", "com", "res", "ind"]),
        "district_code": [10, 2, 10, 7],
        "geometry_id": ["a", "b", "c", "d"],
    })


class TestInferSchema:
    def infer(self, columns, overrides=None) -> TensorSchema:
        return infer_schema("locations", location_frame(), columns, overrides or {})

    @pytest.mark.parametrize(
        ("column", "kind"),
        [
            ("is_coastal", BINARY),
            ("has_station", BINARY),
            ("has_school", BINARY),
            ("pop_density", CONTINUOUS),
            ("num_stops", CONTINUOUS),
            ("zone_class", CATEGORICAL),
            ("land_use_class", CATEGORICAL),
            ("district_code", CONTINUOUS),
        ],
    )
    def test_infers_the_kind_from_dtype_and_values(self, column, kind):
        assert self.infer([column]).columns[0].kind is kind

    def test_categories_are_the_sorted_non_missing_values(self):
        assert self.infer(["zone_class"]).columns[0].categories == ("rural", "urban")
        assert self.infer(["land_use_class"]).columns[0].categories == ("com", "ind", "res")

    def test_an_override_replaces_the_inferred_kind(self):
        schema = self.infer(["num_stops", "district_code"], {"num_stops": COUNT, "district_code": CATEGORICAL})

        assert [column.kind for column in schema.columns] == [COUNT, CATEGORICAL]

    def test_integer_codes_overridden_as_categorical_become_string_categories(self):
        district = self.infer(["district_code"], {"district_code": CATEGORICAL}).columns[0]

        assert district.categories == ("10", "2", "7")

    def test_columns_keep_the_requested_order(self):
        columns = ["pop_density", "is_coastal", "zone_class"]

        assert self.infer(columns).names == columns

    def test_columns_left_out_are_not_in_the_schema(self):
        assert "geometry_id" not in self.infer(["pop_density"]).names

    def test_an_override_for_a_column_outside_the_schema_raises(self):
        with pytest.raises(KeyError, match="pop_densty"):
            self.infer(["pop_density"], {"pop_densty": COUNT})

    def test_a_binary_override_on_other_values_raises(self):
        with pytest.raises(ValueError, match="Binary column `num_stops`"):
            self.infer(["num_stops"], {"num_stops": BINARY})

    def test_a_count_override_on_negative_values_raises(self):
        frame = pd.DataFrame({"net_change": [-1.0, 2.0]})

        with pytest.raises(ValueError, match="Count column `net_change`"):
            infer_schema("locations", frame, ["net_change"], {"net_change": COUNT})

    def test_warns_once_per_inferred_column_and_not_for_overrides(self, caplog):
        with caplog.at_level(logging.WARNING, logger="activitygraphs.schema"):
            self.infer(["pop_density", "is_coastal", "num_stops"], {"num_stops": COUNT})

        warned = [record.getMessage() for record in caplog.records]
        assert len(warned) == 2
        assert any("`locations.pop_density`" in message and "continuous" in message for message in warned)
        assert not any("num_stops" in message for message in warned)
