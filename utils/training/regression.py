"""Regression trainer implementation."""

from typing import Any, Iterable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from .base import BaseTrainer
from ..training_config import TrainingConfig
from ..training_logger import TrainingLogger


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks.

    Uses MSELoss by default and tracks loss as the primary metric.
    Automatically reshapes targets to ``(-1, 1)``.

    Example:
        >>> trainer = RegressionTrainer(model, train_loader, val_loader, config)
        >>> result = trainer.train()
        >>> print(result["loss"], result.get("val_loss"))
    """

    @property
    def default_loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    def prepare_batch(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Reshape y to (-1, 1) for regression."""
        return X.to(self.device), y.to(self.device).reshape(-1, 1)

    def compute_metrics(self, y_hat: Tensor, y: Tensor, loss: float) -> dict[str, float]:
        return {"loss": loss}

    def format_train_metrics(self, metrics: dict[str, float]) -> dict[str, str]:
        return {"loss": f"{metrics['loss']:.4f}"}

    def format_val_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> dict[str, str]:
        return {
            "train": f"{train_metrics['loss']:.4f}",
            "val": f"{val_metrics['val_loss']:.4f}",
        }

    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"train_loss": train_metrics["loss"]}
        if val_metrics:
            result["val_loss"] = val_metrics.get("val_loss")
        return result

    def format_epoch_message(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> str:
        base = f"Epoch {epoch}/{num_epochs} — Loss: {train_metrics['loss']:.4f}"
        if val_metrics:
            return f"{base}, Val Loss: {val_metrics['val_loss']:.4f}"
        return base
