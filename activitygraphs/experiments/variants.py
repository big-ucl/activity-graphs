"""Model variants: an architecture from ``cfg.models`` paired with its input features, and the name it is reported under."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from omegaconf import DictConfig, OmegaConf

from activitygraphs.config import ModelConfig, TwoTowerConfig, VariantConfig
from activitygraphs.models.features import InputFeatures

BIAS_SUFFIXES = {"embedding_mlp": "mlpbias", "feature_mlp": "featbias", "node": "nodebias"}

FEATURE_SUFFIXES = {
    "use_home_pe": "nohpe",
    "use_dist": "nodist",
    "use_demographics": "nodemo",
    "use_rwpe": "norwpe",
}
"""The suffix of each ``InputFeatures`` flag that differs from its default."""


@dataclass(frozen=True)
class Variant:
    """One model of an experiment: an architecture and the input features it is trained with.

    Attributes:
        model: The architecture, its optimiser settings and its hyperparameters, from ``cfg.models``.
        features: The input features the variant is trained with.
    """

    model: ModelConfig
    features: InputFeatures = field(default_factory=InputFeatures)

    @property
    def name(self) -> str:
        """Name of the variant, reported in program outputs.

        The name is composed of (1) the model's ``name``, then (2) the architecture options in ``OPTION_SUFFIXES``, then
        (3) a suffix for every input flag that differs from default.
        """
        parts = [self.model.name]

        if self.model.architecture in OPTION_SUFFIXES:
            parts += OPTION_SUFFIXES[self.model.architecture](self.model)

        for flag in self.features.departures:
            parts.append(FEATURE_SUFFIXES[flag])

        return "-".join(parts)


def with_overrides(model: ModelConfig, overrides: dict) -> ModelConfig:
    """Return a copy of ``model`` with ``overrides`` merged in; an interpolated name follows the new values."""
    return cast(ModelConfig, OmegaConf.merge(model, overrides))


def build_variants(entries: Sequence[VariantConfig], models: dict[str, ModelConfig]) -> list[Variant]:
    """Expand the variant list of an experiment into its variants, one per entry and one per value of a sweep.

    Args:
        entries: The variant list, ``cfg.experiment.variants``.
        models: The architectures the entries are keyed into, ``cfg.models`` with any overfitting overrides.

    Returns:
        The variants, in the order of ``entries``, with a sweep expanded in the order of its values.

    Raises:
        ValueError: If a sweep varies more than one field, if a field set by an entry does not change the variant
            name, or if two variants share a name.
    """
    variants = []
    for entry in entries:
        spec = _validated_entry(entry)
        model = models[spec.model]
        features = InputFeatures(**spec.features)

        for overrides in _entry_overrides(spec):
            _check_overrides_are_named(model, features, overrides)
            variants.append(Variant(with_overrides(model, overrides), features))

    _check_distinct_names(variants)

    return variants


def variant_hyperparams(variant: Variant) -> dict:
    """Collect the architecture config and the input flags of ``variant``, as logged to W&B."""
    features = variant.features

    return {
        "architecture": OmegaConf.to_container(variant.model, resolve=True),
        "use_dist": features.use_dist,
        "use_home_pe": features.use_home_pe,
        "use_demographics": features.use_demographics,
        "use_rwpe": features.use_rwpe,
    }


def _validated_entry(entry: VariantConfig) -> VariantConfig:
    """Return ``entry`` with the defaults of ``VariantConfig`` filled in.

    Raises:
        omegaconf.errors.ConfigKeyError: If ``entry`` has a key that is not a field of ``VariantConfig``.
        omegaconf.errors.MissingMandatoryValue: If ``entry`` has no ``model``.
    """
    return cast(VariantConfig, OmegaConf.merge(OmegaConf.structured(VariantConfig), entry))


def _entry_overrides(spec: VariantConfig) -> list[dict]:
    """Return the model config overrides of every variant of ``spec``: one per sweep value, or ``spec.set`` alone.

    Raises:
        ValueError: If ``spec.sweep`` has more than one field.
    """
    overrides = cast(dict, OmegaConf.to_container(cast(DictConfig, spec.set)))
    sweep = cast(dict, OmegaConf.to_container(cast(DictConfig, spec.sweep)))
    if not sweep:
        return [overrides]
    if len(sweep) > 1:
        raise ValueError(f"A sweep varies one field, got {sorted(sweep)}")

    ((swept_field, values),) = sweep.items()
    swept = []
    for value in values:
        swept.append({**overrides, swept_field: value})

    return swept


def _check_overrides_are_named(model: ModelConfig, features: InputFeatures, overrides: dict) -> None:
    """Check that every field of ``overrides`` that changes ``model`` also changes the variant name.

    Raises:
        ValueError: If a field changes the model but not the name.
    """
    base_name = Variant(model, features).name
    model_values = cast(DictConfig, model)

    for key, value in overrides.items():
        changed = model_values[key] != value
        if changed and Variant(with_overrides(model, {key: value}), features).name == base_name:
            raise ValueError(
                f"Setting `{key}` of `{model.architecture}` leaves the variant name {base_name!r} unchanged. "
                f"Add a suffix for it to OPTION_SUFFIXES or interpolate it in the model's `name`."
            )


def _check_distinct_names(variants: list[Variant]) -> None:
    """Check that no two variants share a name.

    Raises:
        ValueError: If two variants share a name.
    """
    counts = Counter(variant.name for variant in variants)
    repeated = sorted(name for name, count in counts.items() if count > 1)

    if repeated:
        raise ValueError(f"Every variant of an experiment needs its own name, but these repeat: {repeated}")


def _two_tower_suffixes(model: ModelConfig) -> list[str]:
    """Return the suffixes of the two-tower options that differ from their value in ``conf/model/two_tower.yaml``."""
    two_tower = cast(TwoTowerConfig, model)
    suffixes = []

    if two_tower.bias != "location":
        suffixes.append(BIAS_SUFFIXES[two_tower.bias])
    if two_tower.pair_location_features:
        suffixes.append("pairloc")
    if two_tower.node_embedding:
        suffixes.append("nodeemb")
    if two_tower.embedding_dim != 64:
        suffixes.append(f"d{two_tower.embedding_dim}")

    return suffixes


OPTION_SUFFIXES: dict[str, Callable[[ModelConfig], list[str]]] = {"two_tower": _two_tower_suffixes}
"""The function returning the option suffixes of each architecture, keyed by ``architecture``, in name order."""
