"""ActivityDataModule: LightningDataModule wrapping the load_dataset pipeline."""

from pathlib import Path

import lightning as L
import polars as pl
import torch
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.ml.dataset import ActivityDataset, FittedScalers, load_dataset
from activitygraphs.ml.popularity import popularity_logit


def _compute_node_popularity_logit(train_dataset: ActivityDataset) -> torch.Tensor:
    """Compute the per-node visit popularity logit over the training dataset. logit(p_n), where p_n is the
    home-excluded train visit rate of node n. Shape [num_nodes].

    Same per-node rate as VisitFrequencyBaseline, injected into the ML model outputs to see if they learn anything
    beyond the "general" popularity signal."""

    idx = torch.as_tensor(list(train_dataset.indices()), dtype=torch.long)
    labels = train_dataset.spatial_labels[idx].squeeze(-1)
    home_idx = train_dataset.spatial_features[idx, :, train_dataset.is_home_spatial_idx].argmax(dim=1)

    return popularity_logit(labels, home_idx)


def _compute_home_coverage(train_dataset: ActivityDataset, test_dataset: ActivityDataset) -> pl.DataFrame:
    """Compute, for each user in the _test_ dataset, 1) their home node and 2) whether any training user also has a home
    there."""

    # Index of the home node of every user in the survey: [n_users].
    home_node_idx = train_dataset.spatial_features[:, :, train_dataset.is_home_spatial_idx].argmax(dim=1)

    train_idx = torch.as_tensor(list(train_dataset.indices()), dtype=torch.long)
    test_idx = torch.as_tensor(list(test_dataset.indices()), dtype=torch.long)
    train_home_idx = home_node_idx[train_idx].unique().tolist()

    return pl.DataFrame({
        "user_id": test_idx.tolist(),
        "home_node": home_node_idx[test_idx].tolist(),
    }).with_columns(home_seen_in_train=pl.col("home_node").is_in(train_home_idx))


class ActivityDataModule(L.LightningDataModule):
    """LightningDataModule wrapping ``load_dataset`` for activity graph prediction.

    Calls ``load_dataset`` on first ``setup()``, builds PyG DataLoaders, and exposes the
    train-split popularity logits and the home coverage of the test users.

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
        self._pop_logit: torch.Tensor | None = None
        self._home_coverage: pl.DataFrame | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None:
            return

        self._train_dataset, self._val_dataset, self._test_dataset, self._scalers = load_dataset(
            self.cfg, self.val_size, self.test_size, self.seed, self.project_root
        )

        assert self._train_dataset is not None
        assert self._test_dataset is not None

        self._pop_logit = _compute_node_popularity_logit(self._train_dataset)
        self._home_coverage = _compute_home_coverage(self._train_dataset, self._test_dataset)

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
    def pop_logit(self) -> torch.Tensor:
        if self._pop_logit is None:
            raise RuntimeError("Call setup() before accessing pop_logit.")
        return self._pop_logit

    @property
    def home_coverage(self) -> pl.DataFrame:
        if self._home_coverage is None:
            raise RuntimeError("Call setup() before accessing home_coverage.")
        return self._home_coverage
