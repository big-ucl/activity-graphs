"""Top-level experiment runners: experiment definitions, model specification and building and result saving."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning as L
import polars as pl
import torch

from activitygraphs.analysis import check_overfit_health
from activitygraphs.config import Config
from activitygraphs.ml.baselines import (
    ConditionalNodeBaseline,
    GlobalBaseline,
    NodeBaseline,
    UniformBaseline,
)
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.lightning_module import extracted_features_dim
from activitygraphs.ml.losses import Loss, build_loss
from activitygraphs.ml.models import FullyConnectedMLP, GATSkip, GraphTransformer, NodeMLP
from activitygraphs.ml.training import (
    PER_USER_STAGE,
    SCORE_VECTOR_STAGE,
    WandBParams,
    evaluate_baseline,
    train_and_evaluate_model,
)

# =========================================
# Experiments
# =========================================


def comparison_experiment(cfg: Config):
    """Compare the prediction performance of MLP, GATSkip, and GPS variants plus baselines.

    Every model is trained once per seed in ``cfg.train.train_seeds``; the baselines are deterministic,
    so they are evaluated once.

    Results are logged to W&B and written to ``cfg.paths.reports/data/{dataset}-results-{n}.parquet``, with the
    per-user test scores alongside in ``{dataset}-per-user-{n}.parquet``.
    """
    setup = setup_experiment(cfg)
    train_dataset = setup.train_dataset
    hidden_channels = setup.hidden_channels
    dropout = setup.dropout
    lr_for = setup.lr_for

    gat_layers = 8
    gps_layers = 2
    mlp_layers = 3

    gat_name = f"GATSkip-{gat_layers}-res"
    gps_name = f"GTransformer-{gps_layers}-res"

    mlp = partial(build_mlp, train_dataset, mlp_layers, hidden_channels, dropout)
    gat = partial(build_gat, train_dataset, gat_layers, hidden_channels, dropout)
    gps = partial(build_gps, train_dataset, gps_layers, hidden_channels, dropout)

    model_specs: list[ModelSpec] = [
        ModelSpec("MLP", mlp, lr_for("MLP"), {}),
        ModelSpec("MLP-dist", partial(mlp, full_info=True), lr_for("MLP"), {"full_info": True}),
        ModelSpec(
            "MLP-hpe",
            partial(mlp, full_info=True, use_home_pe=True),
            lr_for("MLP"),
            {"full_info": True, "use_home_pe": True},
        ),
        ModelSpec(gat_name, gat, lr_for("GATSkip"), {}),
        ModelSpec(gat_name + "-dist", partial(gat, full_info=True), lr_for("GATSkip"), {"full_info": True}),
        ModelSpec(gat_name + "-hpe", partial(gat, use_home_pe=True), lr_for("GATSkip"), {"use_home_pe": True}),
        ModelSpec(
            gat_name + "-hpe-dist",
            partial(gat, full_info=True, use_home_pe=True),
            lr_for("GATSkip"),
            {"full_info": True, "use_home_pe": True},
        ),
        ModelSpec(gps_name, partial(gps, use_home_pe=True), lr_for("GTransformer"), {"use_home_pe": True}),
        ModelSpec(
            gps_name + "-dist",
            partial(gps, full_info=True, use_home_pe=True),
            lr_for("GTransformer"),
            {"full_info": True, "use_home_pe": True},
        ),
    ]

    model_results = run_model_specs(setup, model_specs)

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results)


def depth_sweep_experiment(cfg: Config):
    """Sweep number of message passing layers of the GATSkip (with HPE and home distance) over ``cfg.train.depths``."""
    setup = setup_experiment(cfg)
    dataset, hidden, dropout = setup.train_dataset, setup.hidden_channels, setup.dropout
    mlp_layers = 3

    model_specs = [
        ModelSpec(
            "MLP-dist",
            partial(build_mlp, dataset, mlp_layers, hidden, dropout, full_info=True),
            setup.lr_for("MLP"),
            {"full_info": True, "extra_hyperparams": {"depth": 0}},
        ),
    ]

    for depth in cfg.train.depths:
        gnn_spec = ModelSpec(
            f"GATSkip-{depth}-res-hpe-dist",
            partial(build_gat, dataset, depth, hidden, dropout, full_info=True, use_home_pe=True),
            setup.lr_for("GATSkip"),
            {"full_info": True, "use_home_pe": True, "extra_hyperparams": {"depth": depth}},
        )

        model_specs.append(gnn_spec)

    model_results = run_model_specs(setup, model_specs)

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results)


def demographics_ablation_experiment(cfg: Config):
    """Run an ablation on the user demographics with the two best performing ML models from ``comparison_experiment``.

    Runs ``use_demographics`` True/False for each retained architecture at every training seed, with
    everything else fixed, so the pair can be compared paired per user. Baselines are evaluated alongside.
    """
    setup = setup_experiment(cfg)
    train_dataset = setup.train_dataset
    hidden_channels = setup.hidden_channels
    dropout = setup.dropout

    gat_layers = 8
    mlp_layers = 3

    builders = {
        "MLP-dist": (partial(build_mlp, train_dataset, mlp_layers, hidden_channels, dropout), setup.lr_for("MLP")),
        f"GATSkip-{gat_layers}-res-dist": (
            partial(build_gat, train_dataset, gat_layers, hidden_channels, dropout),
            setup.lr_for("GATSkip"),
        ),
    }

    model_specs = []

    for name, (builder_f, lr) in builders.items():
        for use_demographics in [True, False]:
            model_spec = ModelSpec(
                f"{name}-{'demo' if use_demographics else 'nodemo'}",
                partial(builder_f, full_info=True, use_demographics=use_demographics),
                lr,
                {"full_info": True, "use_demographics": use_demographics},
            )

            model_specs.append(model_spec)

    model_results = run_model_specs(setup, model_specs)

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *setup.baseline_results)


def overfit_health_experiment(cfg: Config):
    """Check if each architecture can overfit on a single batch.

    Forces ``train.overfit_batches`` and requires ``train.log_train_ranking``, since the check is read entirely off
    ``train_r_precision``. A healthy model drives ``train_r_precision`` to ~1.0 on a batch it has memorised; failure is
    capacity or optimisation rather than data, and invalidates every downstream reading. Runs one seed and no baselines.
    """
    cfg = copy.deepcopy(cfg)
    cfg.train.experiment = "overfit_health"
    cfg.train.log_train_ranking = True
    cfg.train.overfit_batches = max(1, cfg.train.overfit_batches)
    cfg.train.train_seeds = cfg.train.train_seeds[:1]

    setup = setup_experiment(cfg, with_baselines=False)
    train_dataset = setup.train_dataset
    hidden_channels = setup.hidden_channels
    dropout = setup.dropout

    gat_layers = 8
    mlp_layers = 3

    model_specs = [
        ModelSpec(
            "MLP-dist",
            partial(build_mlp, train_dataset, mlp_layers, hidden_channels, dropout, full_info=True),
            setup.lr_for("MLP"),
            {"full_info": True},
        ),
        ModelSpec(
            f"GATSkip-{gat_layers}-res-dist",
            partial(build_gat, train_dataset, gat_layers, hidden_channels, dropout, full_info=True),
            setup.lr_for("GATSkip"),
            {"full_info": True},
        ),
    ]

    model_results = run_model_specs(setup, model_specs)

    if cfg.train.fast_dev_run:
        return

    results = pl.concat(model_results, how="diagonal")
    save_results(cfg.paths.reports, cfg.data.name, results)

    with pl.Config(tbl_rows=-1, float_precision=4):
        print("\n-- overfit health: best train_r_precision on the memorised batch --")
        print(check_overfit_health(results))


# =========================================
# Setup
# =========================================


@dataclass(frozen=True)
class ExperimentSetup:
    """Shared experiment setup: dataset split, loss, baselines, and model hyperparameters.

    Built once per experiment by ``setup_experiment``, so all experiments train against the
    same split, loss, and baselines.
    """

    cfg: Config
    train_dataset: ActivityDataset
    run_model: Callable[..., pl.DataFrame]
    baseline_results: list[pl.DataFrame]
    train_seeds: list[int]
    hidden_channels: int
    dropout: float
    lr_for: Callable[[str], float]


def setup_experiment(cfg: Config, with_baselines: bool = True) -> ExperimentSetup:
    """Setup an experiment. Configure logging, build the datamodule, loss, baselines,
    and a preconfigured ``train_and_evaluate_model`` partial function that will run for each model.

    Hyperparameters not exposed by the config are fixed here: hidden_channels=128, dropout=0.2 and
    the per-model learning rates (all overridden when ``cfg.train.overfit_batches > 0``, unless
    ``cfg.train.overfit_lr`` is null). The split is
    drawn with ``cfg.train.split_seed`` and is never reseeded, so every model in every experiment is
    scored on the same test users.

    ``with_baselines=False`` skips fitting the frequency baselines.
    """
    run_group = f"{cfg.data.name}-{cfg.train.experiment}-{datetime.now():%Y%m%d-%H%M%S}"
    wandb_params = WandBParams(
        use_wandb=cfg.train.wandb,
        project=cfg.train.wandb_project,
        entity=cfg.train.wandb_entity,
        group=run_group,
        dataset_name=cfg.data.name,
        experiment=cfg.train.experiment,
    )

    batch_size = cfg.train.batch_size
    val_size = cfg.train.val_size
    test_size = cfg.train.test_size
    split_seed = cfg.train.split_seed
    train_seeds = list(cfg.train.train_seeds)

    L.seed_everything(split_seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = ActivityDataModule(cfg, val_size=val_size, test_size=test_size, seed=split_seed, batch_size=batch_size)
    datamodule.setup()

    train_dataset = datamodule.train_dataset

    home_hop_distance = torch.as_tensor(train_dataset.home_hop_distance, dtype=torch.float)
    loss = build_loss(cfg.train.loss, home_hop_distance, train_dataset.is_home_col_idx)

    hidden_channels = 128

    overfitting = cfg.train.overfit_batches > 0
    debug = overfitting or cfg.train.debug

    if overfitting:
        dropout = 0.0
        epochs = cfg.train.overfit_epochs
        lr = cfg.train.overfit_lr
        weight_decay = 0.0
    else:
        dropout = 0.2
        epochs = cfg.train.epochs
        lr = 1e-3
        weight_decay = 1e-4

    verbose = 1

    num_nodes = train_dataset[0].num_nodes
    baseline_results = [
        res.with_columns(seed=pl.lit(None, dtype=pl.Int64))
        for res in (
            measure_baselines(num_nodes, datamodule, loss, wandb_params, cfg.train.save_score_vectors)
            if with_baselines
            else []
        )
    ]

    models_dir = cfg.paths.models

    # Per-model learning rates. GPS (GraphTransformer) is not as good as GATSkip at 1e-3. If `overfit`, all models get
    # the same lr=0.01
    lr_by_model = {"MLP": 1e-3, "GATSkip": 1e-3, "GTransformer": 1e-4, "MLP-dist": 1e-3, "FullMLP": 1e-3}

    def lr_for(key: str) -> float:
        """Learning rate for a model. Overfit mode forces one rate unless ``train.overfit_lr`` is null,
        in which case every architecture keeps the rate it is normally trained at."""
        return lr if overfitting and lr is not None else lr_by_model[key]

    my_run_experiment = partial(
        train_and_evaluate_model,
        datamodule=datamodule,
        loss=loss,
        num_epochs=epochs,
        verbose=verbose,
        weight_decay=weight_decay,
        model_save_dir=models_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        overfit_batches=cfg.train.overfit_batches,
        schedule_lr=cfg.train.schedule_lr,
        compile_model=cfg.train.compile,
        wandb_params=wandb_params,
        debug=debug,
        store_score_vectors=cfg.train.save_score_vectors,
        log_train_ranking=cfg.train.log_train_ranking,
    )

    return ExperimentSetup(
        cfg=cfg,
        train_dataset=train_dataset,
        run_model=my_run_experiment,
        baseline_results=baseline_results,
        train_seeds=train_seeds,
        hidden_channels=hidden_channels,
        dropout=dropout,
        lr_for=lr_for,
    )


def measure_baselines(
    num_nodes,
    datamodule: ActivityDataModule,
    loss: Loss,
    wandb_params: WandBParams,
    store_score_vectors: bool = False,
):
    """Fit and evaluate all four frequency baselines; return a list of result dicts."""
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    is_home_idx = datamodule.train_dataset.is_home_col_idx

    uniform_base = UniformBaseline()
    global_base = GlobalBaseline().fit(train_loader)
    node_base = NodeBaseline(num_nodes).fit(train_loader)
    conditional_base = ConditionalNodeBaseline(num_nodes, is_home_idx).fit(train_loader)

    results = []

    for name, baseline in [
        ("Uniform", uniform_base),
        ("GlobalMarginal", global_base),
        ("NodeMarginal", node_base),
        ("ConditionalNodeMarginal", conditional_base),
    ]:
        res = evaluate_baseline(
            baseline,
            datamodule,
            loss,
            name,
            k=datamodule.train_dataset.median_realised_size,
            wandb_params=wandb_params,
            store_score_vectors=store_score_vectors,
        )
        results.append(res)

    return results


# =========================================
# Running models and baselines
# =========================================


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a PyTorch model.

    Model with ``model_name`` is initialised using the ``build_model_f`` callable. Learning rate ``lr`` and other parameters
    ``train_and_eval_kwargs`` kwargs are passed to ``training.train_and_evaluate_model``"""

    model_name: str
    build_model_f: Callable[[], torch.nn.Module]
    lr: float
    train_and_eval_kwargs: dict


def run_model_specs(setup: ExperimentSetup, model_specs: list[ModelSpec]) -> list[pl.DataFrame]:
    """Train every spec once per training seed and return the seed-tagged result frames.

    Only training is reseeded, so the seed spread across repeated runs of one spec measures
    training stochasticity alone rather than split variance.
    """
    results = []

    for train_seed in setup.train_seeds:
        L.seed_everything(train_seed, workers=True)

        for spec in model_specs:
            frame = setup.run_model(
                model=spec.build_model_f(),
                name=spec.model_name,
                lr=spec.lr,
                run_tag=f"seed{train_seed}" if len(setup.train_seeds) > 1 else None,
                **spec.train_and_eval_kwargs,
            )
            results.append(frame.with_columns(seed=pl.lit(train_seed, dtype=pl.Int64)))

    return results


def build_gat(
    dataset: ActivityDataset,
    num_gcn_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    full_info: bool = False,
    use_pop_feature: bool = False,
    use_home_pe: bool = False,
) -> GATSkip:
    """Instantiate a ``GATSkip`` model sized for ``dataset`` (1 pre-layer, 3 post-layers)."""
    edge_dim = dataset[0].edge_attr.size(-1)

    return GATSkip(
        num_pre_layers=1,
        num_gcn_layers=num_gcn_layers,
        num_post_layers=3,
        in_channels=extracted_features_dim(
            dataset,
            use_demographics=use_demographics,
            full_info=full_info,
            use_pop_feature=use_pop_feature,
            use_home_pe=use_home_pe,
        ),
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes - 1,
        edge_dim=edge_dim,
        dropout=dropout,
        residuals=True,
    )


def build_gps(
    dataset: ActivityDataset,
    num_gps_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    full_info: bool = False,
    use_pop_feature: bool = False,
    use_home_pe: bool = False,
):
    """Instantiate a ``GraphTransformer`` (GPS) model sized for ``dataset``."""
    edge_dim = dataset[0].edge_attr.size(-1)

    return GraphTransformer(
        in_channels=extracted_features_dim(
            dataset,
            use_demographics=use_demographics,
            full_info=full_info,
            use_pop_feature=use_pop_feature,
            use_home_pe=use_home_pe,
        ),
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes - 1,
        edge_dim=edge_dim,
        num_layers=num_gps_layers,
        num_heads=4,
        dropout=dropout,
    )


def build_mlp(
    dataset: ActivityDataset,
    mlp_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    full_info: bool = False,
    use_pop_feature: bool = False,
    use_home_pe: bool = False,
) -> NodeMLP:
    """Instantiate a ``NodeMLP`` model sized for ``dataset``.

    With ``full_info=True`` the input width accounts for the appended distance-from-home
    feature (the distance-augmented MLP baseline).
    """
    return NodeMLP(
        mlp_layers,
        in_channels=extracted_features_dim(
            dataset,
            use_demographics=use_demographics,
            full_info=full_info,
            use_pop_feature=use_pop_feature,
            use_home_pe=use_home_pe,
        ),
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes - 1,
        dropout=dropout,
    )


def build_full_mlp(
    dataset: ActivityDataset,
    mlp_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    use_pop_feature: bool = False,
    use_home_pe: bool = False,
) -> FullyConnectedMLP:
    """Instantiate a ``FullyConnectedMLP`` baseline (has all info from all nodes) sized for ``dataset``."""
    return FullyConnectedMLP(
        num_nodes=dataset.num_nodes,
        in_features=extracted_features_dim(
            dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature, use_home_pe=use_home_pe
        ),
        hidden_channels=hidden_channels,
        num_layers=mlp_layers,
        dropout=dropout,
    )


# =========================================
# Saving results
# =========================================


def save_results(path: str | Path, name: str, *results: pl.DataFrame):
    """Concatenate result DataFrames and write them to ``<path>/data/``.

    The aggregate rows (``stage`` of ``fit``/``test``) go to ``<name>-results-<n>.parquet``, the
    per-user summary rows to ``<name>-per-user-<n>.parquet`` and the full per-user score vectors to
    ``<name>-scores-<n>.parquet``. All three are numbered ``n``. They are split because they are
    different sizes, so only they only need to be loaded when required for analysis.
    """
    path = Path(path) / "data"

    ends = (f.stem.split("-")[-1] for f in path.iterdir() if f.suffix == ".parquet" and f.name.startswith(name))
    nums = (int(end) if end.isdecimal() else 0 for end in ends)
    max_num = max([0, *nums])

    combined = pl.concat(results, how="diagonal")
    per_user = combined.filter(pl.col("stage") == PER_USER_STAGE)
    score_vectors = combined.filter(pl.col("stage") == SCORE_VECTOR_STAGE)
    aggregate = combined.filter(~pl.col("stage").is_in([PER_USER_STAGE, SCORE_VECTOR_STAGE]))

    drop_empty_columns(aggregate).write_parquet(path / f"{name}-results-{max_num + 1}.parquet")

    if not per_user.is_empty():
        drop_empty_columns(per_user).write_parquet(path / f"{name}-per-user-{max_num + 1}.parquet")

    if not score_vectors.is_empty():
        drop_empty_columns(score_vectors).write_parquet(path / f"{name}-scores-{max_num + 1}.parquet")


def drop_empty_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that are null for every row (an artefact of the diagonal concat)."""
    if df.is_empty():
        return df

    return df.select([col for col in df.columns if df[col].null_count() < df.height])
