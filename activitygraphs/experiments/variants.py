"""Model variants: an architecture from ``cfg.models`` paired with its input features, and their training loop."""

from dataclasses import dataclass

import lightning as L
import polars as pl
import torch

from activitygraphs.config import ModelConfig
from activitygraphs.datasets import VisitDataset
from activitygraphs.experiments.architectures import ModelSizes, architecture_builder
from activitygraphs.experiments.setup import ExperimentSetup
from activitygraphs.experiments.training import train_and_evaluate_model
from activitygraphs.ml.lightning_module import extracted_features_dim


@dataclass(frozen=True)
class Variant:
    """One model of an experiment: an architecture and the input features it is trained with.

    Attributes:
        model: The architecture, its optimiser settings and its hyperparameters, from ``cfg.models``.
        full_info: Append each node's distance from home to its features.
        use_home_pe: Append the home-anchored positional encoding to each node's features.
        use_demographics: Concatenate the user attributes onto each node's features.
        suffix: Appended to the name after the feature flags.
        extra_hyperparams: Run-specific values to log alongside the standard ones.
    """

    model: ModelConfig
    full_info: bool = False
    use_home_pe: bool = False
    use_demographics: bool = True
    suffix: str | None = None
    extra_hyperparams: dict | None = None

    @property
    def name(self) -> str:
        """Name the variant is reported under: the architecture name, then ``hpe``, ``dist`` and the suffix."""
        parts = [self.model.name]

        if self.use_home_pe:
            parts.append("hpe")
        if self.full_info:
            parts.append("dist")
        if self.suffix is not None:
            parts.append(self.suffix)

        return "-".join(parts)


def build_model(variant: Variant, dataset: VisitDataset) -> torch.nn.Module:
    """Instantiate the model of ``variant``, sized for ``dataset`` and the variant's features."""
    sizes = ModelSizes(
        in_channels=extracted_features_dim(
            dataset,
            use_demographics=variant.use_demographics,
            full_info=variant.full_info,
            use_home_pe=variant.use_home_pe,
        ),
        out_channels=dataset.num_classes - 1,
        edge_dim=dataset.edge_attr.size(-1),
    )

    return architecture_builder(variant.model)(variant.model, sizes)


def run_variants(setup: ExperimentSetup, variants: list[Variant]) -> list[pl.DataFrame]:
    """Train every variant once per training seed and return the seed-tagged result frames.

    Only training is reseeded, so the seed spread across repeated runs of one variant measures
    training stochasticity alone rather than split variance.
    """
    cfg = setup.cfg
    results = []

    for train_seed in setup.train_seeds:
        L.seed_everything(train_seed, workers=True)

        for variant in variants:
            frame = train_and_evaluate_model(
                model=build_model(variant, setup.datamodule.train_dataset),
                datamodule=setup.datamodule,
                loss=setup.loss,
                max_recall_k=cfg.train.max_recall_k,
                recall_ks=list(cfg.analysis.recall_curve_ks),
                num_epochs=setup.num_epochs,
                name=variant.name,
                lr=variant.model.lr,
                weight_decay=variant.model.weight_decay,
                full_info=variant.full_info,
                use_demographics=variant.use_demographics,
                use_home_pe=variant.use_home_pe,
                extra_hyperparams=variant.extra_hyperparams,
                model_save_dir=cfg.paths.models,
                fast_dev_run=cfg.train.fast_dev_run,
                overfit_batches=cfg.train.overfit_batches,
                schedule_lr=cfg.train.schedule_lr,
                compile_model=cfg.train.compile,
                store_score_vectors=cfg.train.save_score_vectors,
                log_train_ranking=cfg.train.log_train_ranking,
                wandb_params=setup.wandb_params,
                debug=setup.debug,
                run_tag=f"seed{train_seed}" if len(setup.train_seeds) > 1 else None,
            )
            results.append(frame.with_columns(seed=pl.lit(train_seed, dtype=pl.Int64)))

    return results
