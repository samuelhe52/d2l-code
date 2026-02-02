"""Training utilities for regression tasks."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from ..io import save_model
from ..logging import TrainingLogger
from ..training_config import TrainingConfig, resolve_training_config

def train(model: nn.Module, dataloader: Iterable, num_epochs: Optional[int] = None, lr: Optional[float] = None,
          loss_fn: Optional[nn.Module] = None, optimizer: Optional[Optimizer] = None, save_path: Optional[str] = None, 
          verbose: bool = True, logger: TrainingLogger | None = None, val_dataloader: Optional[Iterable] = None,
          device: Optional[torch.device] = None, config: TrainingConfig | None = None) -> Tuple[float | None, float | None]:
    """Train a regression model.

    Args:
        model: PyTorch model to train.
        dataloader: Iterable of training batches ``(X, y)``.
        num_epochs: Number of epochs (overrides ``config`` if provided).
        lr: Learning rate (overrides ``config`` if provided).
        loss_fn: Loss function (default: ``MSELoss``).
        optimizer: Optimizer (default: ``SGD`` with the provided ``lr``).
        save_path: Optional path to save model parameters after training.
        verbose: Whether to print per-epoch metrics.
        logger: Optional ``TrainingLogger`` to record metrics.
        val_dataloader: Optional validation iterable for per-epoch eval.
        device: Torch device; inferred if ``None``.
        config: Optional ``TrainingConfig``; explicit args take precedence.

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
