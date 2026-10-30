"""Unit tests for the per-column transforms fitted on the training split."""

import logging

import torch

from activitygraphs.datasets import transforms as transforms_module
from activitygraphs.datasets.transforms import (
    TRANSFORM_POLICY,
    FeatureTransforms,
    GroupTransform,
    Transform,
    TransformPolicy,
)
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema
from activitygraphs.tensors.build import DISTANCE_SCHEMA, EDGE_SCHEMA, USER_NODE_SCHEMA

CONTINUOUS = ColumnKind.CONTINUOUS
COUNT = ColumnKind.COUNT
BINARY = ColumnKind.BINARY
CATEGORICAL = ColumnKind.CATEGORICAL

PASSTHROUGH = Transform.PASSTHROUGH
ZSCORE = Transform.ZSCORE
LOG1P_ZSCORE = Transform.LOG1P_ZSCORE


def one_column_per_kind() -> TensorSchema:
    return TensorSchema(
        "locations",
        (
            ColumnSpec("density", CONTINUOUS),
            ColumnSpec("distance", COUNT),
            ColumnSpec("is_flag", BINARY),
            ColumnSpec("band", CATEGORICAL, ("low", "mid", "high")),
        ),
    )


def one_column_per_kind_values(num_rows: int = 50) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    density = torch.rand(num_rows, generator=generator) * 100.0
    distance = torch.rand(num_rows, generator=generator) * 20_000.0
    is_flag = (torch.rand(num_rows, generator=generator) > 0.5).float()
    band = torch.randint(0, 4, (num_rows,), generator=generator).float()

    return torch.stack([density, distance, is_flag, band], dim=1)


def zscore(column: torch.Tensor) -> torch.Tensor:
    return (column - column.mean()) / column.std(unbiased=False)


class TestGroupTransform:
    def test_each_kind_gets_its_default_transform(self):
        schema = one_column_per_kind()
        values = one_column_per_kind_values()
        transforms = (ZSCORE, LOG1P_ZSCORE, PASSTHROUGH, PASSTHROUGH)

        out = GroupTransform.fit(schema, transforms, values).apply(values)

        torch.testing.assert_close(out[:, 0], zscore(values[:, 0]))
        torch.testing.assert_close(out[:, 1], zscore(torch.log1p(values[:, 1])))
        torch.testing.assert_close(out[:, 2], values[:, 2], rtol=0, atol=0)
        torch.testing.assert_close(out[:, 3], values[:, 3], rtol=0, atol=0)

    def test_keeps_the_shape_of_per_user_tensors(self):
        schema = TensorSchema("distances", (ColumnSpec("home_distance", COUNT),))
        values = torch.rand(4, 7, 1) * 1000.0

        out = GroupTransform.fit(schema, (LOG1P_ZSCORE,), values).apply(values)

        assert out.shape == values.shape
        torch.testing.assert_close(out.flatten(), zscore(torch.log1p(values.flatten())))

    def test_leaves_the_input_tensor_unchanged(self):
        schema = one_column_per_kind()
        values = one_column_per_kind_values()
        original = values.clone()

        GroupTransform.fit(schema, (ZSCORE, LOG1P_ZSCORE, PASSTHROUGH, PASSTHROUGH), values).apply(values)

        torch.testing.assert_close(values, original, rtol=0, atol=0)

    def test_fits_no_scaler_when_every_column_passes_through(self):
        values = torch.tensor([[0.0], [1.0], [0.0]])

        fitted = GroupTransform.fit(USER_NODE_SCHEMA, (PASSTHROUGH,), values)

        assert fitted.scaler is None
        torch.testing.assert_close(fitted.apply(values), values, rtol=0, atol=0)

    def test_applies_the_statistics_of_the_fitted_rows(self):
        schema = TensorSchema("user_attributes", (ColumnSpec("age", CONTINUOUS),))
        train = torch.tensor([[10.0], [30.0]])

        out = GroupTransform.fit(schema, (ZSCORE,), train).apply(torch.tensor([[20.0], [50.0]]))

        torch.testing.assert_close(out, torch.tensor([[0.0], [3.0]]))

    def test_warns_about_a_constant_zscored_column(self, caplog):
        schema = TensorSchema("user_attributes", (ColumnSpec("constant", BINARY),))
        values = torch.ones(5, 1)

        with caplog.at_level(logging.WARNING):
            out = GroupTransform.fit(schema, (ZSCORE,), values).apply(values)

        assert "user_attributes.constant" in caplog.text
        torch.testing.assert_close(out, torch.zeros(5, 1))


class TestTransformPolicy:
    def test_is_home_passes_through(self):
        assert TRANSFORM_POLICY.column_transforms(USER_NODE_SCHEMA) == (PASSTHROUGH,)

    def test_binary_demographics_pass_through(self):
        schema = TensorSchema(
            "user_attributes", (ColumnSpec("is_flag", BINARY), ColumnSpec("hh_num_adults", CONTINUOUS))
        )

        assert TRANSFORM_POLICY.column_transforms(schema) == (PASSTHROUGH, ZSCORE)

    def test_distances_are_log1p_then_zscored(self):
        assert TRANSFORM_POLICY.column_transforms(DISTANCE_SCHEMA) == (LOG1P_ZSCORE,)

    def test_edge_weights_are_zscored_without_log1p(self):
        assert TRANSFORM_POLICY.column_transforms(EDGE_SCHEMA) == (ZSCORE,)

    def test_a_group_without_overrides_follows_the_kind(self):
        assert TRANSFORM_POLICY.column_transforms(one_column_per_kind()) == (
            ZSCORE,
            LOG1P_ZSCORE,
            PASSTHROUGH,
            PASSTHROUGH,
        )


class TestFeatureTransforms:
    def make(self, fingerprint: dict) -> tuple[FeatureTransforms, torch.Tensor]:
        schema = one_column_per_kind()
        values = one_column_per_kind_values()
        transforms = TRANSFORM_POLICY.column_transforms(schema)
        group = GroupTransform.fit(schema, transforms, values)

        return FeatureTransforms({"locations": group}, TRANSFORM_POLICY, fingerprint), values

    def test_round_trips_through_save_and_read(self, tmp_path):
        fitted, values = self.make({"n_users": 3})
        path = tmp_path / "transforms.pkl"

        fitted.save(path)
        read = FeatureTransforms.read(path)

        assert read.is_current({"n_users": 3})
        torch.testing.assert_close(read.groups["locations"].apply(values), fitted.groups["locations"].apply(values))

    def test_is_stale_for_another_fingerprint(self):
        fitted, _ = self.make({"n_users": 3})

        assert not fitted.is_current({"n_users": 4})

    def test_is_stale_once_the_policy_changes(self, monkeypatch):
        fitted, _ = self.make({"n_users": 3})
        policy = TransformPolicy(TRANSFORM_POLICY.kinds, group_overrides={})
        monkeypatch.setattr(transforms_module, "TRANSFORM_POLICY", policy)

        assert not fitted.is_current({"n_users": 3})
