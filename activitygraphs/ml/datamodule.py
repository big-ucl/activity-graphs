"""ActivityDataModule: LightningDataModule wrapping the load_dataset pipeline."""

from pathlib import Path

import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.ml.dataset import ActivityDataset, FittedScalers, load_dataset


def _compute_training_weights(train_dataset: ActivityDataset) -> torch.Tensor:
    """Compute BCE positive-class weight as sqrt(neg_count / pos_count) over the training dataset."""
    idx = train_dataset.indices()
    y = train_dataset.spatial_labels[idx]
    num_pos = y.sum()
    return torch.sqrt((y.numel() - num_pos) / num_pos)


def _compute_node_popularity_logit(train_dataset: ActivityDataset) -> torch.Tensor:
    """Compute the per-node visit popularity logit over the training dataset. logit(p_n), where p_n is the
    train visit rate of node n. Shape [num_nodes].

    Same per-node rate as NodeBaseline, injected into the ML model outputs to see if they learn anything beyond the
    "general" popularity signal."""

    idx = train_dataset.indices()
    y = train_dataset.spatial_labels[idx].float()
    p = y.mean(dim=0).squeeze(-1).clamp(1e-6, 1 - 1e-6)

    return torch.log(p / (1 - p))


class ActivityDataModule(L.LightningDataModule):
    """LightningDataModule wrapping ``load_dataset`` for activity graph prediction.

    Calls ``load_dataset`` on first ``setup()``, builds PyG DataLoaders, and exposes the
    computed positive-class weight needed to initialise ``ActivityGraphModule``.

    Args:
        cfg: Hydra config containing dataset paths.
        val_size: Fraction of individuals reserved for validation.
        test_size: Fraction of individuals reserved for testing.
        seed: Random seed for the train/validation split.
        batch_size: Number of graphs per DataLoader batch.
        project_root: Optional override for the project root path.
    """

    def __init__(
        self,
        cfg: Config,
        val_size: float,
        test_size: float,
        seed: int,
        batch_size: int,
        project_root: Path | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.project_root = project_root

        self._train_dataset: ActivityDataset | None = None
        self._val_dataset: ActivityDataset | None = None
        self._test_dataset: ActivityDataset | None = None
        self._scalers: FittedScalers | None = None
        self._pos_weight: torch.Tensor | None = None
        self._pop_logit: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        self._train_dataset, self._val_dataset, self._test_dataset, self._scalers = load_dataset(
            self.cfg, self.val_size, self.test_size, self.seed, self.project_root
        )

        assert self._train_dataset is not None

        self._pos_weight = _compute_training_weights(self._train_dataset)
        self._pop_logit = _compute_node_popularity_logit(self._train_dataset)

    def train_dataloader(self) -> pyg.loader.DataLoader:
        assert self._train_dataset is not None
        return pyg.loader.DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> pyg.loader.DataLoader:
        assert self._val_dataset is not None
        return pyg.loader.DataLoader(self._val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self) -> pyg.loader.DataLoader:
        assert self._test_dataset is not None
        return pyg.loader.DataLoader(self._test_dataset, batch_size=self.batch_size, num_workers=8)

    @property
    def train_dataset(self) -> ActivityDataset:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> ActivityDataset:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> ActivityDataset:
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    @property
    def scalers(self) -> FittedScalers:
        if self._scalers is None:
            raise RuntimeError("Call setup() before accessing scalers.")
        return self._scalers

    @property
    def pos_weight(self) -> torch.Tensor:
        if self._pos_weight is None:
            raise RuntimeError("Call setup() before accessing pos_weight.")
        return self._pos_weight

    @property
    def pop_logit(self) -> torch.Tensor:
        if self._pop_logit is None:
            raise RuntimeError("Call setup() before accessing pop_logit.")
        return self._pop_logit
