"""ActivityDataModule: LightningDataModule wrapping the load_dataset pipeline."""

from pathlib import Path

import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.ml.dataset import ActivityDataset, FittedScalers, load_dataset


class ActivityDataModule(L.LightningDataModule):
    """LightningDataModule wrapping ``load_dataset`` for activity graph prediction.

    Calls ``load_dataset`` on first ``setup()``, builds PyG DataLoaders, and exposes the
    computed positive-class weight needed to initialise ``ActivityGraphModule``.

    Args:
        cfg: Hydra config containing dataset paths.
        test_size: Fraction of individuals reserved for validation.
        seed: Random seed for the train/validation split.
        batch_size: Number of graphs per DataLoader batch.
        project_root: Optional override for the project root path.
    """

    def __init__(
        self,
        cfg: Config,
        test_size: float,
        seed: int,
        batch_size: int,
        project_root: Path | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.project_root = project_root

        self._train_dataset: ActivityDataset | None = None
        self._val_dataset: ActivityDataset | None = None
        self._scalers: FittedScalers | None = None
        self._pos_weight: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        from activitygraphs.ml.experiment import compute_training_weights

        self._train_dataset, self._val_dataset, self._scalers = load_dataset(
            self.cfg, self.test_size, self.seed, self.project_root
        )
        train_loader = pyg.loader.DataLoader(self._train_dataset, batch_size=self.batch_size)
        self._pos_weight = compute_training_weights(train_loader)

    def train_dataloader(self) -> pyg.loader.DataLoader:
        return pyg.loader.DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> pyg.loader.DataLoader:
        return pyg.loader.DataLoader(self._val_dataset, batch_size=self.batch_size)

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
    def scalers(self) -> FittedScalers:
        if self._scalers is None:
            raise RuntimeError("Call setup() before accessing scalers.")
        return self._scalers

    @property
    def pos_weight(self) -> torch.Tensor:
        if self._pos_weight is None:
            raise RuntimeError("Call setup() before accessing pos_weight.")
        return self._pos_weight
