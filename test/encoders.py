"""Pass-through feature encoders sized for the synthetic batches of the tests."""

from activitygraphs.ml.features import Demographics, FeatureEncoder, FeatureSource, LocationFeatures, UserNodeFeatures
from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema


def continuous_schema(group: str, width: int) -> TensorSchema:
    """Return the schema of ``width`` continuous columns ``c0``, ``c1``, ... of ``group``."""
    columns = tuple(ColumnSpec(f"c{i}", ColumnKind.CONTINUOUS) for i in range(width))
    return TensorSchema(group, columns)


def make_encoder(
    num_location_features: int, num_user_node_features: int, num_demographics: int, *extra_sources: FeatureSource
) -> FeatureEncoder:
    """Return the encoder of the location and user-node features, then ``extra_sources``, then the demographics.

    The demographics are left out when ``num_demographics`` is 0.
    """
    sources: list[FeatureSource] = [
        LocationFeatures(continuous_schema("locations", num_location_features)),
        UserNodeFeatures(continuous_schema("user_nodes", num_user_node_features)),
        *extra_sources,
    ]
    if num_demographics > 0:
        sources.append(Demographics(continuous_schema("user_attributes", num_demographics)))

    return FeatureEncoder(sources)
