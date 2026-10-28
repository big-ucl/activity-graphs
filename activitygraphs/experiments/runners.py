"""Top-level experiment runners: the experiments ``experiment.name`` selects, each a list of variants and baselines."""

import logging
from typing import cast

import lightning as L
import polars as pl

from activitygraphs.analysis import check_overfit_health
from activitygraphs.config import Config, DepthSweepConfig
from activitygraphs.experiments.results import save_results
from activitygraphs.experiments.setup import ExperimentSetup, setup_experiment, with_overrides
from activitygraphs.experiments.training import train_and_evaluate_model
from activitygraphs.experiments.variants import Variant, build_feature_encoder, build_model

log = logging.getLogger(__name__)


def run_variants(setup: ExperimentSetup, variants: list[Variant]) -> list[pl.DataFrame]:
    """Train every variant once per training seed and return the seed-tagged result frames.

    Only training is reseeded, so the seed spread across repeated runs of one variant measures
    training stochasticity alone rather than split variance.
    """
    cfg = setup.cfg
    dataset = setup.datamodule.train_dataset
    results = []

    for train_seed in setup.train_seeds:
        L.seed_everything(train_seed, workers=True)

        for variant in variants:
            encoder = build_feature_encoder(variant, dataset, cfg.train.home_pe)
            model = build_model(variant, dataset, encoder.out_dim)
            run_tag = f"seed{train_seed}" if len(setup.train_seeds) > 1 else None

            result_df = train_and_evaluate_model(
                model=model,
                feature_encoder=encoder,
                variant=variant,
                datamodule=setup.datamodule,
                loss=setup.loss,
                cfg=cfg,
                run_dir=setup.run_dir,
                wandb_params=setup.wandb_params,
                debug=setup.debug,
                run_tag=run_tag,
            )
            results.append(result_df.with_columns(seed=pl.lit(train_seed, dtype=pl.Int64)))

    return results


def comparison_experiment(cfg: Config):
    """Compare the prediction performance of MLP, GAT, and GPS variants plus baselines.

    Every model is trained once per seed in ``cfg.train.train_seeds``; the baselines are deterministic,
    so they are evaluated once.

    Results are logged to W&B and written to ``results.parquet`` in the run directory, with the
    per-user test scores alongside in ``per-user.parquet``.
    """
    setup = setup_experiment(cfg)
    mlp, gat, gps = setup.models["mlp"], setup.models["gat"], setup.models["gps"]

    variants = [
        Variant(mlp),
        Variant(mlp, full_info=True),
        Variant(mlp, full_info=True, use_home_pe=True),
        Variant(gat),
        Variant(gat, full_info=True),
        Variant(gat, use_home_pe=True),
        Variant(gat, full_info=True, use_home_pe=True),
        Variant(gps, use_home_pe=True),
        Variant(gps, full_info=True, use_home_pe=True),
    ]

    model_results = run_variants(setup, variants)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)


def depth_sweep_experiment(cfg: Config):
    """Sweep number of message passing layers of the GAT (with HPE and home distance) over ``cfg.experiment.depths``."""
    setup = setup_experiment(cfg)
    gat = setup.models["gat"]

    variants = [Variant(setup.models["mlp"], full_info=True, extra_hyperparams={"depth": 0})]

    for depth in cast(DepthSweepConfig, cfg.experiment).depths:
        gat_at_depth = with_overrides(gat, {"num_gcn_layers": depth})
        variants.append(Variant(gat_at_depth, full_info=True, use_home_pe=True, extra_hyperparams={"depth": depth}))

    model_results = run_variants(setup, variants)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)


def demographics_ablation_experiment(cfg: Config):
    """Run an ablation on the user demographics with the two best performing ML models from ``comparison_experiment``.

    Runs ``use_demographics`` True/False for each retained architecture at every training seed, with
    everything else fixed, so the pair can be compared paired per user. Baselines are evaluated alongside.
    """
    setup = setup_experiment(cfg)

    variants = [
        Variant(
            model,
            full_info=True,
            use_demographics=use_demographics,
            suffix="demo" if use_demographics else "nodemo",
        )
        for model in (setup.models["mlp"], setup.models["gat"])
        for use_demographics in [True, False]
    ]

    model_results = run_variants(setup, variants)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)


def core_experiment(cfg: Config):
    """Compare the core variants beside the baselines, the short run for paired comparisons between code changes.

    Trains ``MLP-dist`` with and without the random-walk positional encoding, ``MLP-hpe-dist``, and the GAT and GPS
    with the same features as ``MLP-hpe-dist``, every one once per training seed. The RWPE-free MLP is the control
    for whether graph structure helps beyond node features.
    """
    setup = setup_experiment(cfg)
    mlp, gat, gps = setup.models["mlp"], setup.models["gat"], setup.models["gps"]

    variants = [
        Variant(mlp, full_info=True, use_rwpe=False),
        Variant(mlp, full_info=True),
        Variant(mlp, full_info=True, use_home_pe=True),
        Variant(gat, full_info=True, use_home_pe=True),
        Variant(gps, full_info=True, use_home_pe=True),
    ]

    model_results = run_variants(setup, variants)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *model_results, *setup.baseline_results, setup.home_distances)


def overfit_health_experiment(cfg: Config):
    """Check if each architecture can overfit on a single batch.

    Requires ``trainer.overfit_batches > 0`` and ``train.log_train_ranking``, both set by
    ``conf/experiment/overfit_health.yaml``, since the check is read entirely off ``train_avg_recall@K``. A healthy
    model drives ``train_avg_recall@K`` to ~1.0 on a batch it has memorised; failure is capacity or optimisation rather
    than data, and invalidates every downstream reading. Runs no baselines.

    Raises:
        ValueError: ``trainer.overfit_batches`` is 0 or ``train.log_train_ranking`` is off.
    """
    if cfg.trainer.overfit_batches == 0 or not cfg.train.log_train_ranking:
        raise ValueError("overfit_health needs trainer.overfit_batches > 0 and train.log_train_ranking=true.")

    setup = setup_experiment(cfg, with_baselines=False)

    variants = [
        Variant(setup.models["mlp"], full_info=True),
        Variant(setup.models["gat"], full_info=True),
    ]

    model_results = run_variants(setup, variants)

    if setup.run_dir is None:
        return

    results = pl.concat(model_results, how="diagonal")
    save_results(setup.run_dir, results, setup.home_distances)

    with pl.Config(tbl_rows=-1, float_precision=4):
        health = str(check_overfit_health(results, cfg.train.max_recall_k))

    log.info("Overfit health: best train_avg_recall@%d on the memorised batch\n%s", cfg.train.max_recall_k, health)


def baselines_experiment(cfg: Config):
    """Fit and evaluate the baselines alone, with no learned models.

    The split is fixed by ``cfg.train.split_seed``, so the per-user rows pair with any learned run of the same dataset.
    """
    setup = setup_experiment(cfg)

    if setup.run_dir is None:
        return

    save_results(setup.run_dir, *setup.baseline_results, setup.home_distances)
