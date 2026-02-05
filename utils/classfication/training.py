"""Training utilities for classification tasks.

This module provides backward-compatible ``train`` and ``validate`` functions.
For new code, consider using ``ClassificationTrainer`` directly::

    from utils.training import ClassificationTrainer

    trainer = ClassificationTrainer(model, train_loader, val_loader, config)
    trainer.train()
"""

from typing import Iterable, Optional, overload

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from ..training_config import TrainingConfig
from ..training_logger import TrainingLogger
from ..training import ClassificationTrainer, get_device
from ..training.classification import accuracy


# Re-export accuracy for backward compatibility
__all__ = ["train", "validate", "accuracy"]


@overload
def train(
    model: nn.Module,
    dataloader: Iterable,
    val_dataloader: Iterable | None,
    config: TrainingConfig,
) -> None: ...


@overload
def train(
    model: nn.Module,
    dataloader: Iterable,
    val_dataloader: Iterable | None = None,
    *,
    num_epochs: int,
    lr: float,
    loss_fn: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    save_path: str | None = None,
    verbose: bool = True,
    logger: TrainingLogger | None = None,
    device: torch.device | None = None,
) -> None: ...


def train(
    model: nn.Module,
    dataloader: Iterable,
    val_dataloader: Iterable | None = None,
    config: TrainingConfig | None = None,
    *,
    num_epochs: int | None = None,
    lr: float | None = None,
    loss_fn: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    save_path: str | None = None,
    verbose: bool = True,
    logger: TrainingLogger | None = None,
    device: torch.device | None = None,
) -> None:
    """Train a classification model.

    This function supports two usage patterns:

    **Preferred (config-based):**
        >>> train(model, train_loader, val_loader, config)

    **Legacy (explicit params):**
        >>> train(model, train_loader, num_epochs=10, lr=0.01, ...)

    Args:
        model: PyTorch model to train.
        dataloader: Iterable of training batches ``(X, y)``.
        val_dataloader: Optional validation iterable for per-epoch eval.
        config: Training configuration (preferred). If provided, other args
            override the corresponding config values.
        num_epochs: Number of epochs (overrides ``config`` if provided).
        lr: Learning rate (overrides ``config`` if provided).
        loss_fn: Loss function (default: ``CrossEntropyLoss``).
        optimizer: Optimizer (default: ``SGD`` with the provided ``lr``).
        save_path: Optional path to save model parameters after training.
        verbose: Whether to print per-epoch metrics.
        logger: Optional ``TrainingLogger`` to record metrics.
        device: Torch device; inferred if ``None``.
    """
    trainer = ClassificationTrainer(
        model,
        dataloader,
        val_dataloader,
        config,
        num_epochs=num_epochs,
        lr=lr,
        loss_fn=loss_fn,
        optimizer=optimizer,
        save_path=save_path,
        verbose=verbose,
        logger=logger,
        device=device,
    )
    trainer.train()


def validate(
    model: nn.Module,
    dataloader: Iterable,
    device: Optional[torch.device] = None,
    loss_fn: Optional[nn.Module] = None,
) -> tuple[float, float | None]:
    """Evaluate the model on a validation dataset.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Iterable of validation batches ``(X, y)``.
        device: Torch device; inferred if ``None``.
        loss_fn: Loss function for computing validation loss (optional).

    Returns:
        Tuple of (validation accuracy, validation loss). Loss is None if loss_fn not provided.
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    accuracies = []
    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            accuracies.append(accuracy(y_hat, y))
            if loss_fn is not None:
                losses.append(loss_fn(y_hat, y).item())

    avg_acc = sum(accuracies) / len(accuracies)
    avg_loss = sum(losses) / len(losses) if losses else None
    return avg_acc, avg_loss
