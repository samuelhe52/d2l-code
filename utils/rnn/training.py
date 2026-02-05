"""Training utilities for RNN models.

This module provides backward-compatible ``train`` and ``validate`` functions.
For new code, consider using ``RNNTrainer`` directly::

    from utils.training import RNNTrainer

    trainer = RNNTrainer(model, train_loader, val_loader, config)
    trainer.train()
"""

from typing import Iterable, Optional, overload

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from ..training_config import TrainingConfig
from ..training_logger import TrainingLogger
from ..training import RNNTrainer, get_device
from ..training.rnn import perplexity


# Re-export perplexity for backward compatibility
__all__ = ["train", "validate", "perplexity"]


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
    grad_clip: float | None = None,
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
    grad_clip: float | None = None,
) -> None:
    """Train an RNN model.

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
        grad_clip: Max norm for gradient clipping. If ``None``, no clipping.
    """
    trainer = RNNTrainer(
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
        grad_clip=grad_clip,
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
        Tuple of (validation perplexity, validation loss). Loss is None if loss_fn not provided.
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    ppls = []
    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat, _ = model(X)
            ppls.append(perplexity(y_hat, y))
            if loss_fn is not None:
                losses.append(loss_fn(y_hat, y).item())

    avg_ppl = sum(ppls) / len(ppls)
    avg_loss = sum(losses) / len(losses) if losses else None
    return avg_ppl, avg_loss
