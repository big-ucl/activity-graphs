"""Unit tests for the categorical codes and the schema-driven encoder of a tensor group."""

import polars as pl
import pytest
import torch

from activitygraphs.ml.schema_encoder import SchemaEncoder, default_emb_dim
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.tensors.build import encode_categories

INCOME_BAND = ColumnSpec("income_band", ColumnKind.CATEGORICAL, ("low", "mid", "high"))
VEHICLES = ColumnSpec("hh_num_vehicles", ColumnKind.CATEGORICAL, ("0", "1", "2"))


def mixed_schema() -> TensorSchema:
    return TensorSchema(
        "user_attributes",
        (
            ColumnSpec("age", ColumnKind.CONTINUOUS),
            INCOME_BAND,
            ColumnSpec("can_drive", ColumnKind.BINARY),
            VEHICLES,
        ),
    )


class TestEncodeCategories:
    def test_a_category_gets_its_position_plus_one(self):
        values = pl.Series(["high", "low", "mid", "low"])

        assert encode_categories(values, INCOME_BAND).tolist() == [3, 1, 2, 1]

    def test_missing_and_unknown_values_get_code_zero(self):
        values = pl.Series(["mid", None, "very high"])

        assert encode_categories(values, INCOME_BAND).tolist() == [2, 0, 0]

    def test_integer_values_match_their_string_categories(self):
        values = pl.Series([2, 0, None, 7], dtype=pl.Int64)

        assert encode_categories(values, VEHICLES).tolist() == [3, 1, 0, 0]

    def test_codes_are_int64(self):
        assert encode_categories(pl.Series(["low"]), INCOME_BAND).dtype == torch.long


class TestSchemaEncoder:
    @pytest.mark.parametrize(("cardinality", "width"), [(2, 1), (4, 2), (11, 6), (31, 16), (100, 16)])
    def test_default_embedding_width(self, cardinality, width):
        assert default_emb_dim(cardinality) == width

    def test_out_dim_is_the_numeric_columns_plus_the_embedding_widths(self):
        encoder = SchemaEncoder(mixed_schema())

        assert encoder.out_dim == 2 + default_emb_dim(4) + default_emb_dim(4)

    def test_numeric_columns_come_first_unchanged_then_one_embedding_per_categorical_column(self):
        encoder = SchemaEncoder(mixed_schema(), emb_dim=lambda cardinality: 3)
        x_num = torch.tensor([[35.0, 1.0], [70.0, 0.0]])
        x_cat = torch.tensor([[3, 0], [1, 2]])

        out = encoder(x_num, x_cat)

        assert out.shape == (2, encoder.out_dim)
        torch.testing.assert_close(out[:, :2], x_num, rtol=0, atol=0)
        torch.testing.assert_close(out[:, 2:5], encoder.embeddings[0].weight[x_cat[:, 0]])
        torch.testing.assert_close(out[:, 5:8], encoder.embeddings[1].weight[x_cat[:, 1]])

    def test_without_categorical_columns_the_numeric_columns_pass_through(self):
        schema = TensorSchema("user_attributes", (ColumnSpec("age", ColumnKind.CONTINUOUS),))
        encoder = SchemaEncoder(schema)
        x_num = torch.tensor([[35.0], [70.0]])

        out = encoder(x_num, torch.zeros(2, 0, dtype=torch.long))

        assert encoder.out_dim == 1
        assert not list(encoder.parameters())
        torch.testing.assert_close(out, x_num, rtol=0, atol=0)

    def test_the_embeddings_are_trained(self):
        encoder = SchemaEncoder(mixed_schema())
        x_cat = torch.tensor([[1, 2]])

        encoder(torch.zeros(1, 2), x_cat).sum().backward()

        assert encoder.embeddings[0].weight.grad is not None
        assert encoder.embeddings[0].weight.grad[1].abs().sum() > 0
