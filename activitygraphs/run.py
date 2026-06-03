"""Top-level experiment runners: model builders, baseline evaluation, and result persistence."""

import functools
from datetime import datetime
from pathlib import Path

import polars as pl

from activitygraphs.config import Config
from activitygraphs.ml.baselines import (
    ConditionalNodeBaseline,
    GlobalBaseline,
    NodeBaseline,
    UniformBaseline,
)
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.experiment import evaluate_baseline, run_experiment, WandBParams
from activitygraphs.ml.lightning_module import extracted_features_dim
from activitygraphs.ml.models import FullyConnectedMLP, GATSkip, GraphTransformer, NodeMLP


def build_gat(
    dataset: ActivityDataset,
    num_gcn_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    use_pop_feature: bool = False,
) -> GATSkip:
    """Instantiate a ``GATSkip`` model sized for ``dataset`` (1 pre-layer, 3 post-layers)."""
    edge_dim = dataset[0].edge_attr.size(-1)

    return GATSkip(
        num_pre_layers=1,
        num_gcn_layers=num_gcn_layers,
        num_post_layers=3,
        in_channels=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
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
    use_pop_feature: bool = False,
):
    """Instantiate a ``GraphTransformer`` (GPS) model sized for ``dataset``."""
    edge_dim = dataset[0].edge_attr.size(-1)

    return GraphTransformer(
        in_channels=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
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
) -> NodeMLP:
    """Instantiate a ``NodeMLP`` model sized for ``dataset``.

    With ``full_info=True`` the input width accounts for the appended distance-from-home
    feature (the distance-augmented MLP baseline).
    """
    return NodeMLP(
        mlp_layers,
        in_channels=extracted_features_dim(
            dataset, use_demographics=use_demographics, full_info=full_info, use_pop_feature=use_pop_feature
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
) -> FullyConnectedMLP:
    """Instantiate a ``FullyConnectedMLP`` baseline (has all info from all nodes) sized for ``dataset``."""
    return FullyConnectedMLP(
        num_nodes=dataset.num_nodes,
        in_features=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
        hidden_channels=hidden_channels,
        num_layers=mlp_layers,
        dropout=dropout,
    )


def save_results(path: str | Path, name: str, *results: pl.DataFrame):
    """Concatenate result DataFrames and write to ``<path>/data/<name>-results.parquet``."""
    path = Path(path) / "data"

    ends = (f.stem.split("-")[-1] for f in path.iterdir() if f.suffix == ".parquet" and f.name.startswith(name))
    nums = (int(end) if end.isdecimal() else 0 for end in ends)
    max_num = max([0, *nums])

    pl.concat(results, how="diagonal").write_parquet(path / f"{name}-results-{max_num + 1}.parquet")


def measure_baselines(num_nodes, datamodule: ActivityDataModule, wandb_params: WandBParams):
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
            baseline, datamodule, name, k=datamodule.train_dataset.median_realised_size, wandb_params=wandb_params
        )
        results.append(res)

    return results


def comparison_experiment(cfg: Config):
    """Compare the prediction performance of MLP, GATSkip, GPS (with and without L1) plus baselines.

    Fixed hyperparameters: batch_size=64, epochs=50, hidden_channels=128, dropout=0.2.
    Results are written to ``cfg.paths.reports/data/{dataset}-results-{n}.parquet``.
    """

    run_group = f"{cfg.data.name}-{datetime.now():%Y%m%d-%H%M%S}"
    wandb_params = WandBParams(
        use_wandb=cfg.train.wandb,
        project=cfg.train.wandb_project,
        entity=cfg.train.wandb_entity,
        group=run_group,
        dataset_name=cfg.data.name,
    )

    batch_size = 64
    val_size = 0.1
    test_size = 0.2
    seed = 42

    datamodule = ActivityDataModule(cfg, val_size=val_size, test_size=test_size, seed=seed, batch_size=batch_size)
    datamodule.setup()

    train_dataset = datamodule.train_dataset

    hidden_channels = 128
    gat_layers = 8
    gps_layers = 2
    mlp_layers = 3

    overfitting = cfg.train.overfit_batches > 0
    debug = overfitting or cfg.train.debug

    if overfitting:
        dropout = 0.0
        epochs = 500
        lr = 1e-2
        weight_decay = 0.0
    else:
        dropout = 0.2
        epochs = cfg.train.epochs
        lr = 1e-3
        weight_decay = 1e-4

    verbose = 1

    num_nodes = train_dataset[0].num_nodes
    baseline_results = measure_baselines(num_nodes, datamodule, wandb_params)

    models_dir = cfg.paths.models

    # Per-model learning rates. GPS (GraphTransformer) is not as good as GATSkip at 1e-3. If `overfit`, all models get
    # the same lr=0.01
    lr_by_model = {"MLP": 1e-3, "GATSkip": 1e-3, "GTransformer": 1e-4, "MLP-dist": 1e-3, "FullMLP": 1e-3}

    def lr_for(key: str) -> float:
        return lr if overfitting else lr_by_model[key]

    # Regularise full model more to avoid overfitting.
    full_dropout = 0.0 if overfitting else 0.5
    full_weight_decay = weight_decay if overfitting else 1e-3

    gat_name = f"GATSkip-{gat_layers}-res"
    gps_name = f"GTransformer-{gps_layers}-res"

    mlp = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    mlp_pop_off = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    mlp_pop_feat = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout, use_pop_feature=True)

    gat = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    gat_pop_off = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    gat_pop_feat = build_gat(train_dataset, gat_layers, hidden_channels, dropout, use_pop_feature=True)

    full_mlp = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)
    full_mlp_off = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)
    full_mlp_feat = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout, use_pop_feature=True)

    # gps = build_gps(train_dataset, gps_layers, hidden_channels, dropout)
    # mlp_dist = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout, full_info=True)
    # full_mlp = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)

    my_run_experiment = functools.partial(
        run_experiment,
        datamodule=datamodule,
        num_epochs=epochs,
        verbose=verbose,
        weight_decay=weight_decay,
        model_save_dir=models_dir,
        fast_dev_run=cfg.train.fast_dev_run,
        overfit_batches=cfg.train.overfit_batches,
        schedule_lr=cfg.train.schedule_lr,
        wandb_params=wandb_params,
        debug=debug,
    )

    results_mlp = my_run_experiment(model=mlp, name="MLP", lr=lr_for("MLP"))
    results_mlp_off = my_run_experiment(model=mlp_pop_off, name="MLP-off", lr=lr_for("MLP"), pop_mode="offset")
    results_mlp_feat = my_run_experiment(model=mlp_pop_feat, name="MLP-feat", lr=lr_for("MLP"), pop_mode="feature")

    results_full = my_run_experiment(
        model=full_mlp, name="FullMLP", lr=lr_for("FullMLP"), weight_decay=full_weight_decay
    )
    results_full_off = my_run_experiment(
        model=full_mlp_off,
        name="FullMLP-pop-offset",
        lr=lr_for("FullMLP"),
        weight_decay=full_weight_decay,
        pop_mode="offset",
    )
    results_full_feat = my_run_experiment(
        model=full_mlp_feat,
        name="FullMLP-pop-feat",
        lr=lr_for("FullMLP"),
        weight_decay=full_weight_decay,
        pop_mode="feature",
    )

    results_gat = my_run_experiment(model=gat, name=gat_name, lr=lr_for("GATSkip"))
    results_gat_off = my_run_experiment(
        model=gat_pop_off, name=gat_name + "-pop-offset", lr=lr_for("GATSkip"), pop_mode="offset"
    )
    results_gat_feat = my_run_experiment(
        model=gat_pop_feat, name=gat_name + "-pop-feat", lr=lr_for("GATSkip"), pop_mode="feature"
    )

    # results_gps = my_run_experiment(model=gps, name=gps_name, lr=lr_for("GTransformer"))
    # results_mlp_dist = my_run_experiment(model=mlp_dist, name="MLP-dist", lr=lr_for("MLP-dist"), full_info=True)

    # model_results = [results_mlp, results_gat, results_gps, results_mlp_dist, results_full]

    model_results = [
        results_mlp,
        results_mlp_off,
        results_mlp_feat,
        results_gat,
        results_gat_off,
        results_gat_feat,
        results_full,
        results_full_off,
        results_full_feat,
    ]

    if cfg.train.fast_dev_run:
        return

    save_results(cfg.paths.reports, cfg.data.name, *model_results, *baseline_results)
