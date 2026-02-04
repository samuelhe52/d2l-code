"""Training utilities for regression tasks."""

from typing import Iterable, Optional, Tuple, overload

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from ..io import save_model
from ..training_logger import TrainingLogger
from ..training_config import TrainingConfig, resolve_training_config


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
    cfg = resolve_training_config(
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

    device = cfg.device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model.to(device)

    loss_fn = cfg.loss_fn or nn.MSELoss()
    if cfg.optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        optimizer = cfg.optimizer
    
    epoch_pbar = tqdm(range(cfg.num_epochs), desc='Training', unit='epoch')
    last_train_loss: float | None = None
    last_val_loss: float | None = None
    for epoch in epoch_pbar:
        model.train()  # Ensure model is in training mode
        losses = []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{cfg.num_epochs}', leave=False)
        for X, y in batch_pbar:
            X = X.to(device)
            y = y.to(device).reshape(-1, 1)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            batch_pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = sum(losses) / len(losses)
        last_train_loss = avg_loss
        
        # Evaluate on validation set if provided
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(model, loss_fn, val_dataloader, device=device)
            last_val_loss = val_loss
            epoch_pbar.set_postfix(train=f'{avg_loss:.4f}', val=f'{val_loss:.4f}')
            if cfg.verbose:
                tqdm.write(f'Epoch {epoch + 1}/{cfg.num_epochs} — Loss: {avg_loss:.4f}, '
                        f'Val Loss: {val_loss:.4f}')
        else:
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}')
            if cfg.verbose:
                tqdm.write(f'Epoch {epoch + 1}/{cfg.num_epochs} — Loss: {avg_loss:.4f}')
        
        if cfg.logger is not None:
            cfg.logger.log_epoch(epoch, train_loss=avg_loss, val_loss=val_loss)
    
    if cfg.save_path is not None:
        save_model(model, cfg.save_path)

    return last_train_loss, last_val_loss


def validate(model: nn.Module, loss_fn: nn.Module, dataloader: Iterable, device: Optional[torch.device] = None) -> float:
    """Evaluate the model on a validation dataset.

    Args:
        model: PyTorch model to evaluate.
        loss_fn: Loss function used for evaluation.
        dataloader: Iterable of validation batches ``(X, y)``.
        device: Torch device; inferred if ``None``.

    Returns:
        Average validation loss.
    """
    # Infer device if not explicitly provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

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
    avg_loss = sum(losses) / len(losses)
    return avg_loss
