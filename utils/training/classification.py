"""Classification trainer implementation."""

from typing import Any, Iterable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from .base import BaseTrainer
from .config import TrainingConfig
from .logger import TrainingLogger


def accuracy(y_hat: Tensor, y: Tensor) -> float:
    """Compute the number of correct predictions.

    Args:
        y_hat: Predicted logits of shape ``(batch_size, num_classes)``.
        y: Ground-truth labels of shape ``(batch_size,)``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    preds = y_hat.argmax(dim=1)
    return (preds == y).type(torch.float).sum().item() / len(y)


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification tasks.

    Uses CrossEntropyLoss by default and tracks accuracy as the primary metric.

    Example:
        >>> trainer = ClassificationTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """

    @property
    def default_loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def compute_metrics(self, y_hat: Tensor, y: Tensor, loss: float) -> dict[str, float]:
        return {
            "loss": loss,
            "acc": accuracy(y_hat, y),
        }

    def format_train_metrics(self, metrics: dict[str, float]) -> dict[str, str]:
        return {
            "loss": f"{metrics['loss']:.4f}",
            "acc": f"{metrics['acc']:.4f}",
        }

    def format_val_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> dict[str, str]:
        return {
            "loss": f"{train_metrics['loss']:.4f}",
            "train": f"{train_metrics['acc']:.2%}",
            "val": f"{val_metrics['val_acc']:.2%}",
        }

    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
        }
        if val_metrics:
            result["val_loss"] = val_metrics.get("val_loss")
            result["val_acc"] = val_metrics.get("val_acc")
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
            return f"{base}, Train: {train_metrics['acc']:.2%}, Val: {val_metrics['val_acc']:.2%}"
        return f"{base}, Acc: {train_metrics['acc']:.2%}"
