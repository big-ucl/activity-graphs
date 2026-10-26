"""Top-level experiment runners: the experiments ``train.experiment`` selects, each a list of variants and baselines."""

import copy

import polars as pl

from activitygraphs.analysis import check_overfit_health
from activitygraphs.config import Config
from activitygraphs.experiments.results import save_results
from activitygraphs.experiments.setup import setup_experiment, with_overrides
from activitygraphs.experiments.variants import Variant, run_variants


def comparison_experiment(cfg: Config):
    """Compare the prediction performance of MLP, GATSkip, and GPS variants plus baselines.

    Every model is trained once per seed in ``cfg.train.train_seeds``; the baselines are deterministic,
    so they are evaluated once.

    Results are logged to W&B and written to ``cfg.paths.reports/data/{dataset}-results-{n}.parquet``, with the
    per-user test scores alongside in ``{dataset}-per-user-{n}.parquet``.
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

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results, setup.home_distances)


def depth_sweep_experiment(cfg: Config):
    """Sweep number of message passing layers of the GATSkip (with HPE and home distance) over ``cfg.train.depths``."""
    setup = setup_experiment(cfg)
    gat = setup.models["gat"]

    variants = [Variant(setup.models["mlp"], full_info=True, extra_hyperparams={"depth": 0})]

    for depth in cfg.train.depths:
        gat_at_depth = with_overrides(gat, {"num_gcn_layers": depth})
        variants.append(Variant(gat_at_depth, full_info=True, use_home_pe=True, extra_hyperparams={"depth": depth}))

    model_results = run_variants(setup, variants)

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results, setup.home_distances)


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

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results, setup.home_distances)


def overfit_health_experiment(cfg: Config):
    """Check if each architecture can overfit on a single batch.

    Forces ``train.overfit_batches`` and requires ``train.log_train_ranking``, since the check is read entirely off
    ``train_avg_recall@K``. A healthy model drives ``train_avg_recall@K`` to ~1.0 on a batch it has memorised; failure
    is capacity or optimisation rather than data, and invalidates every downstream reading. Runs one seed and no
    baselines.
    """
    cfg = copy.deepcopy(cfg)
    cfg.train.experiment = "overfit_health"
    cfg.train.log_train_ranking = True
    cfg.train.overfit_batches = max(1, cfg.train.overfit_batches)
    cfg.train.train_seeds = cfg.train.train_seeds[:1]

    setup = setup_experiment(cfg, with_baselines=False)

    variants = [
        Variant(setup.models["mlp"], full_info=True),
        Variant(setup.models["gat"], full_info=True),
    ]

    model_results = run_variants(setup, variants)

    if cfg.train.fast_dev_run:
        return

    results = pl.concat(model_results, how="diagonal")
    save_results(cfg.paths.reports, cfg.data.name, results, setup.home_distances)

    with pl.Config(tbl_rows=-1, float_precision=4):
        print(f"\n-- overfit health: best train_avg_recall@{cfg.train.max_recall_k} on the memorised batch --")
        print(check_overfit_health(results, cfg.train.max_recall_k))


def baselines_experiment(cfg: Config):
    """Fit and evaluate the baselines alone, with no learned models.

    The split is fixed by ``cfg.train.split_seed``, so the per-user rows pair with any learned run of the same dataset.
    """
    setup = setup_experiment(cfg)

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *setup.baseline_results, setup.home_distances)
