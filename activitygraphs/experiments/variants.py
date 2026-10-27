"""Model variants: an architecture from ``cfg.models`` paired with its input features."""

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from activitygraphs.config import ModelConfig
from activitygraphs.datasets import VisitDataset
from activitygraphs.experiments.architectures import ModelSizes, architecture_builder
from activitygraphs.ml.lightning_module import extracted_features_dim


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


def build_model(variant: Variant, dataset: VisitDataset, home_pe_bins: int) -> torch.nn.Module:
    """Instantiate the model of ``variant``, sized for ``dataset`` and the variant's features.

    Args:
        variant: The variant to build.
        dataset: The dataset the model is trained on.
        home_pe_bins: Number of columns of the home-anchored positional encoding, read when the variant uses it.
    """
    sizes = ModelSizes(
        in_channels=extracted_features_dim(
            dataset,
            use_demographics=variant.use_demographics,
            full_info=variant.full_info,
            use_home_pe=variant.use_home_pe,
            n_bins=home_pe_bins,
            use_rwpe=variant.use_rwpe,
        ),
        out_channels=dataset.num_classes - 1,
        edge_dim=dataset.edge_attr.size(-1),
    )

    return architecture_builder(variant.model)(variant.model, sizes)


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
