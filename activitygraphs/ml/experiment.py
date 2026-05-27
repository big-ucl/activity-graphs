"""Training loop, evaluation functions, and ``run_experiment`` orchestrator."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.logging import log

from activitygraphs.ml.loss import LossFn
from activitygraphs.ml.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k


def compute_training_weights(loader: pyg.loader.DataLoader) -> torch.Tensor:
    """Compute BCE positive-class weight as sqrt(neg_count / pos_count) over the full loader."""
    num_neg = torch.tensor(0, dtype=torch.float)
    num_pos = torch.tensor(0, dtype=torch.float)

    for batch in loader:
        num_neg += (batch.y == 0).sum()
        num_pos += batch.y.sum()

    weights = num_neg / num_pos
    return torch.sqrt(weights)


def extract_features(batch: pyg.data.Data | pyg.data.Batch, full_info: bool):
    """Return node features from ``batch.x``, optionally augmented with home features and distances."""
    if not full_info:
        return batch.x

    distances = batch.distances

    if batch.batch is not None:
        home_feature = batch.home_feature[batch.batch].unsqueeze(1)
    else:
        home_feature = torch.full((batch.x.shape[0], 1), batch.home_feature.item())

    return torch.cat([batch.x, home_feature, distances], dim=1)


def train(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    optimizer: torch.optim.Optimizer,
    full_info: bool,
    pos_weight: torch.Tensor,
    loss_fn: LossFn,
    reg: str = None,
    lambda_reg: float = 0.01,
):
    """Run one training epoch and return the mean per-node loss.

    Args:
        device: Device to run on.
        model: Model to train.
        loader: Training data loader.
        optimizer: Optimizer instance (gradients are zeroed and stepped internally).
        full_info: If True, augment features with home info and distances via ``extract_features``.
        pos_weight: Positive-class weight tensor for BCE loss.
        loss_fn: Loss function conforming to ``LossFn`` protocol.
        reg: Optional regularisation type; currently only ``"l1"`` is supported.
        lambda_reg: L1 regularisation coefficient (ignored when ``reg`` is None).

    Returns:
        Mean per-node training loss for the epoch.
    """
    model.train()

    epoch_loss = 0.0
    num_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)

        optimizer.zero_grad()
        out = model(x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out, batch.y.float(), pos_weight=pos_weight)

        if reg == "l1":
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += lambda_reg * l1_norm

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * batch.num_nodes
        num_nodes += batch.num_nodes

        del out
        del loss

    return epoch_loss / num_nodes


@torch.no_grad()
def evaluate_bce(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    full_info: bool,
    pos_weight: torch.Tensor | None = None,
):
    """Compute mean per-node BCE loss over the loader without gradients."""
    model.eval()

    epoch_loss = 0.0
    num_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)

        out = model(x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=pos_weight)

        epoch_loss += loss.item() * batch.num_nodes
        num_nodes += batch.num_nodes

    return epoch_loss / num_nodes


@torch.no_grad()
def evaluate_at_k(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    full_info: bool,
    k: int = 5,
):
    """Return a dict of mean precision@k, recall@k, MRR, and NDCG@k over all graphs in the loader."""
    model.eval()
    precisions, recalls, mrrs, ndcgs = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)
        out = model(x, batch.edge_index, batch.edge_attr, batch.batch)

        for i in range(batch.num_graphs):
            mask = batch.batch == i
            scores = out[mask].squeeze()
            labels = batch.y[mask].squeeze()

            if labels.sum().int().item() == 0:
                continue

            precisions.append(precision_at_k(scores, labels, k))
            recalls.append(recall_at_k(scores, labels, k))
            mrrs.append(mean_reciprocal_rank(scores, labels))
            ndcgs.append(ndcg_at_k(scores, labels, k))

    return {
        f"precision@{k}": sum(precisions) / len(precisions),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
    }


@torch.no_grad()
def evaluate_baseline(
    baseline: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    name: str,
    full_info: bool = False,
    pos_weight: torch.Tensor = None,
):
    """Evaluate a baseline model and return a results dict matching the ``run_experiment`` format."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    baseline = baseline.to(device)
    loss = evaluate_bce(device, baseline, loader, full_info)
    loss_weight = evaluate_bce(device, baseline, loader, full_info, pos_weight=pos_weight)
    metrics = evaluate_at_k(device, baseline, loader, full_info)

    log(
        Model=name,
        bce=loss,
        w_bce=loss_weight,
        precision_at_5=metrics["precision@5"],
        recall_at_5=metrics["recall@5"],
        mrr=metrics["mrr"],
        ndcg_at_5=metrics["ndcg@5"],
    )

    return {
        "name": name,
        "epoch": [0],
        "train_bce": [0.0],
        "train_eval_bce": [0.0],
        "bce": [loss],
        "bce_weight": [loss_weight],
        "precision@5": [metrics["precision@5"]],
        "recall@5": [metrics["recall@5"]],
        "mrr": [metrics["mrr"]],
        "ndcg@5": [metrics["ndcg@5"]],
    }


def run_experiment(
    model: torch.nn.Module,
    train_loader: pyg.loader.DataLoader,
    test_loader: pyg.loader.DataLoader,
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    reg: str = None,
    full_info: bool = False,
    model_save_dir: Path | None = None,
):
    """Train a model and return per-epoch metrics as a dict.

    Uses Adam with weight decay 1e-4 and ReduceLROnPlateau scheduling.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        test_loader: Test data loader for evaluation.
        num_epochs: Number of training epochs.
        verbose: Log every ``verbose`` epochs; 0 disables mid-training logging.
        name: Name used for logging and the saved checkpoint filename.
        lr: Initial learning rate.
        reg: Optional regularisation; ``"l1"`` adds L1 weight penalty.
        full_info: Passed to ``extract_features`` to optionally include home/distance features.
        save: If True, save the final model state dict to ``models/<name>.pth``.

    Returns:
        Dict with keys ``name``, ``epoch``, ``train_bce``, ``train_eval_bce``,
        ``bce``, ``bce_weight``, ``precision@5``, ``recall@5``, ``mrr``, ``ndcg@5``.
    """
    name = name or model.__class__.__name__

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    epoch = train_loss = train_eval_loss = test_loss = test_loss_weight = metrics = None
    train_losses = []
    train_eval_losses = []
    test_losses = []
    test_losses_weighted = []
    metric_losses = {
        "precision@5": [],
        "recall@5": [],
        "mrr": [],
        "ndcg@5": [],
    }

    # noinspection PyTypeChecker
    loss_fn: LossFn = F.binary_cross_entropy_with_logits
    pos_weight = compute_training_weights(train_loader).to(device)

    log(Model=name)

    for epoch in range(1, num_epochs + 1):
        train_loss = train(device, model, train_loader, optimizer, full_info, pos_weight, loss_fn, reg=reg)
        train_eval_loss = evaluate_bce(device, model, train_loader, full_info)
        test_loss = evaluate_bce(device, model, test_loader, full_info)
        test_loss_weight = evaluate_bce(device, model, test_loader, full_info, pos_weight=pos_weight)

        metrics = evaluate_at_k(device, model, test_loader, full_info)

        scheduler.step(test_loss)

        train_losses.append(train_loss)
        train_eval_losses.append(train_eval_loss)
        test_losses.append(test_loss)
        test_losses_weighted.append(test_loss_weight)

        for metric, values in metric_losses.items():
            values.append(metrics[metric])

        if verbose and (epoch - 1) % verbose == 0:
            log(
                Epoch=epoch,
                train_loss=train_loss,
                train_eval_bce=train_eval_loss,
                bce=test_loss,
                w_bce=test_loss_weight,
                precision_at_5=metrics["precision@5"],
                recall_at_5=metrics["recall@5"],
                mrr=metrics["mrr"],
                ndcg_at_5=metrics["ndcg@5"],
            )

    log(
        Epoch=epoch,
        train_loss=train_loss,
        train_eval_bce=train_eval_loss,
        bce=test_loss,
        w_bce=test_loss_weight,
        precision_at_5=metrics["precision@5"],
        recall_at_5=metrics["recall@5"],
        mrr=metrics["mrr"],
        ndcg_at_5=metrics["ndcg@5"],
    )

    if model_save_dir is not None:
        torch.save(model.state_dict(), model_save_dir / f"{name}.pth")

    return {
        "name": name,
        "epoch": list(range(1, num_epochs + 1)),
        "train_bce": train_losses,
        "train_eval_bce": train_eval_losses,
        "bce": test_losses,
        "bce_weight": test_losses_weighted,
        "precision@5": metric_losses["precision@5"],
        "recall@5": metric_losses["recall@5"],
        "mrr": metric_losses["mrr"],
        "ndcg@5": metric_losses["ndcg@5"],
    }
