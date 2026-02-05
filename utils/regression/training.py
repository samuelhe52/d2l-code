"""Training utilities for regression tasks.

This module provides backward-compatible ``train`` and ``validate`` functions.
For new code, consider using ``RegressionTrainer`` directly::

    from utils.training import RegressionTrainer

    trainer = RegressionTrainer(model, train_loader, val_loader, config)
    result = trainer.train()
"""

from typing import Iterable, Optional, Tuple, overload

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from ..training_config import TrainingConfig
from ..training_logger import TrainingLogger
from ..training import RegressionTrainer, get_device


__all__ = ["train", "validate"]


@overload
def train(
    model: nn.Module,
    dataloader: Iterable,
    val_dataloader: Iterable | None,
    config: TrainingConfig,
) -> Tuple[float | None, float | None]: ...


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
) -> Tuple[float | None, float | None]: ...


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
) -> Tuple[float | None, float | None]:
    """Train a regression model.

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
        loss_fn: Loss function (default: ``MSELoss``).
        optimizer: Optimizer (default: ``SGD`` with the provided ``lr``).
        save_path: Optional path to save model parameters after training.
        verbose: Whether to print per-epoch metrics.
        logger: Optional ``TrainingLogger`` to record metrics.
        device: Torch device; inferred if ``None``.

    Returns:
        Tuple of ``(train_loss, val_loss)`` for the final epoch. ``val_loss`` is
        ``None`` when no validation dataloader is provided.
    """
    trainer = RegressionTrainer(
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
    result = trainer.train()

    if result is None:
        return None, None
    return result.get("loss"), result.get("val_loss")


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: Iterable,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate the model on a validation dataset.

    Args:
        model: PyTorch model to evaluate.
        loss_fn: Loss function used for evaluation.
        dataloader: Iterable of validation batches ``(X, y)``.
        device: Torch device; inferred if ``None``.

    Returns:
        Average validation loss.
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).reshape(-1, 1)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss.item())

    return sum(losses) / len(losses)
