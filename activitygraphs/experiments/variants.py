"""Model variants: an architecture from ``cfg.models`` paired with its input features."""

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from activitygraphs.config import HomePEConfig, ModelConfig
from activitygraphs.datasets import Locations
from activitygraphs.models.architectures import ModelSizes, lookup_architecture
from activitygraphs.models.features import (
    Demographics,
    FeatureEncoder,
    FeatureSource,
    HomeAnchoredPE,
    HomeDistance,
    LocationFeatures,
    RandomWalkPE,
    UserNodeFeatures,
)
from activitygraphs.models.scorers import Scorer
from activitygraphs.schema import DatasetSchema


@dataclass(frozen=True)
class Variant:
    """One model of an experiment: an architecture and the input features it is trained with.

    Attributes:
        model: The architecture, its optimiser settings and its hyperparameters, from ``cfg.models``.
        full_info: Append each node's distance from home to its features.
        use_home_pe: Append the home-anchored positional encoding to each node's features.
        use_demographics: Concatenate the user attributes onto each node's features.
        use_rwpe: Append the random-walk positional encoding to each node's features.
        suffix: Appended to the name after the feature flags.
        extra_hyperparams: Run-specific values to log alongside the standard ones.
    """

    model: ModelConfig
    full_info: bool = False
    use_home_pe: bool = False
    use_demographics: bool = True
    use_rwpe: bool = True
    suffix: str | None = None
    extra_hyperparams: dict | None = None

    @property
    def name(self) -> str:
        """Name the variant is reported under: the architecture name, then ``hpe``, ``dist``, ``norwpe`` and the suffix."""
        parts = [self.model.name]

        if self.use_home_pe:
            parts.append("hpe")
        if self.full_info:
            parts.append("dist")
        if not self.use_rwpe:
            parts.append("norwpe")
        if self.suffix is not None:
            parts.append(self.suffix)

        return "-".join(parts)


def build_feature_encoder(
    variant: Variant, locations: Locations, schema: DatasetSchema, home_pe_cfg: HomePEConfig
) -> FeatureEncoder:
    """Instantiate the input ``FeatureEncoder`` for the associated ``variant`` and the survey ``locations``.

    The column order goes:
      1. location features,
      2. per-user node features,
      3. random-walk PE,
      4. home-anchored PE,
      5. demographics, and
      6. distance from home.

    Args:
        variant: The variant whose input sources are encoded.
        locations: The location graph the model is trained on.
        schema: The column schema of each tensor group.
        home_pe_cfg: Settings of the home-anchored positional encoding, read when the variant uses it.
    """
    sources: list[FeatureSource] = [
        LocationFeatures(schema.locations),
        UserNodeFeatures(schema.user_nodes),
    ]

    if variant.use_rwpe:
        sources.append(RandomWalkPE(locations.num_rwpe_features))
    if variant.use_home_pe:
        sources.append(HomeAnchoredPE(home_pe_cfg.bins, home_pe_cfg.max_hops))
    if variant.use_demographics:
        sources.append(Demographics(schema.user_attributes))
    if variant.full_info:
        sources.append(HomeDistance(schema.distances))

    return FeatureEncoder(sources)


def build_model(variant: Variant, schema: DatasetSchema, in_channels: int) -> torch.nn.Module:
    """Instantiate the network of ``variant``, sized for the tensors of ``schema`` and its input width.

    Args:
        variant: The variant to build.
        schema: The column schema of each tensor group.
        in_channels: Width of each node's model input, the ``out_dim`` of the variant's ``FeatureEncoder``.
    """
    sizes = ModelSizes(
        in_channels=in_channels,
        out_channels=schema.targets.width,
        edge_dim=schema.edges.width,
    )

    return lookup_architecture(variant.model).build(variant.model, sizes)


def build_scorer(variant: Variant, locations: Locations, schema: DatasetSchema, home_pe_cfg: HomePEConfig) -> Scorer:
    """Instantiate the scorer of ``variant``: its ``FeatureEncoder`` and its network, run by its architecture's scorer.

    Args:
        variant: The variant to build.
        locations: The location graph the model is trained on.
        schema: The column schema of each tensor group.
        home_pe_cfg: Settings of the home-anchored positional encoding, read when the variant uses it.
    """
    encoder = build_feature_encoder(variant, locations, schema, home_pe_cfg)
    network = build_model(variant, schema, encoder.out_dim)

    return lookup_architecture(variant.model).scorer(encoder, network)


def variant_hyperparams(variant: Variant) -> dict:
    """Collect the architecture config, input flags and extra hyperparameters of ``variant``, as logged to W&B."""
    return {
        "architecture": OmegaConf.to_container(variant.model, resolve=True),
        "full_info": variant.full_info,
        "use_home_pe": variant.use_home_pe,
        "use_demographics": variant.use_demographics,
        "use_rwpe": variant.use_rwpe,
        **(variant.extra_hyperparams or {}),
    }
