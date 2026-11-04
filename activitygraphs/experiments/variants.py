"""Model variants: an architecture from ``cfg.models`` paired with its input features."""

from dataclasses import dataclass, field

from omegaconf import OmegaConf

from activitygraphs.config import ModelConfig
from activitygraphs.models.features import InputFeatures


@dataclass(frozen=True)
class Variant:
    """One model of an experiment: an architecture and the input features it is trained with.

    Attributes:
        model: The architecture, its optimiser settings and its hyperparameters, from ``cfg.models``.
        features: The input features the variant is trained with.
        suffix: Appended to the name after the feature flags.
        extra_hyperparams: Run-specific values to log alongside the standard ones.
    """

    model: ModelConfig
    features: InputFeatures = field(default_factory=InputFeatures)
    suffix: str | None = None
    extra_hyperparams: dict | None = None

    @property
    def name(self) -> str:
        """Name the variant is reported under: the architecture name, then ``hpe``, ``dist``, ``norwpe`` and the suffix."""
        parts = [self.model.name]
        features = self.features

        if features.use_home_pe:
            parts.append("hpe")
        if features.full_info:
            parts.append("dist")
        if not features.use_rwpe:
            parts.append("norwpe")
        if self.suffix is not None:
            parts.append(self.suffix)

        return "-".join(parts)


def variant_hyperparams(variant: Variant) -> dict:
    """Collect the architecture config, input flags and extra hyperparameters of ``variant``, as logged to W&B."""
    features = variant.features

    return {
        "architecture": OmegaConf.to_container(variant.model, resolve=True),
        "full_info": features.full_info,
        "use_home_pe": features.use_home_pe,
        "use_demographics": features.use_demographics,
        "use_rwpe": features.use_rwpe,
        **(variant.extra_hyperparams or {}),
    }
