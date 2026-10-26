"""In-memory Lightning logger collecting a run's metrics as stage-tagged rows."""

from argparse import Namespace
from typing import Any

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class MetricsLogger(Logger):
    """Keeps every logged ``train_*`` / ``val_*`` metric by epoch and every ``test_*`` metric, in memory.

    Passed to the ``Trainer`` beside the W&B or CSV logger, so it receives each metric once Lightning has computed
    its epoch value, after every epoch-end hook has logged.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fit: dict[int, dict[str, float]] = {}
        self._test: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Name of the logger."""
        return "metrics"

    @property
    def version(self) -> int:
        """Version of the logger."""
        return 0

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        """Ignore the hyperparameters, which the run's config already records."""

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Keep the ``train_*`` / ``val_*`` metrics under their ``epoch`` and the ``test_*`` metrics."""
        fit = {key: value for key, value in metrics.items() if key.startswith(("train_", "val_"))}
        test = {key: value for key, value in metrics.items() if key.startswith("test_")}

        if fit:
            self._fit.setdefault(int(metrics["epoch"]), {}).update(fit)
        self._test.update(test)

    @property
    def rows(self) -> list[dict]:
        """Return one ``stage="fit"`` row per epoch, then one ``stage="test"`` row when a test ran."""
        rows: list[dict] = [{"stage": "fit", "epoch": epoch, **metrics} for epoch, metrics in sorted(self._fit.items())]
        if self._test:
            rows.append({"stage": "test", "epoch": None, **self._test})

        return rows
