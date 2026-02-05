"""Training utilities for RNN models."""

from typing import Iterable, Optional, overload

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..io import save_model
from ..training_logger import TrainingLogger
from ..training_config import TrainingConfig, resolve_training_config

def perplexity(y_hat: Tensor, y: Tensor) -> float:
    """Compute perplexity given model predictions and true labels.

    Args:
        y_hat: Predicted logits of shape (batch_size, vocab_size, seq_len).
        y: True labels of shape (batch_size, seq_len).
    Returns:
        Perplexity as a float.
    """
    cross_entropy = nn.CrossEntropyLoss()(y_hat, y).item()
    return torch.exp(torch.tensor(cross_entropy)).item()

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
        >>> train(model, train_loader, config=cfg, val_dataloader=val_loader)

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
        grad_clip=grad_clip,
    )

    device = cfg.device
    # Infer device if not explicitly provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model.to(device)
    loss_fn = cfg.loss_fn or nn.CrossEntropyLoss()
    optimizer = cfg.optimizer or torch.optim.SGD(model.parameters(), lr=cfg.lr)
    
    epoch_pbar = tqdm(range(cfg.num_epochs), desc='Training', unit='epoch')
    for epoch in epoch_pbar:
        model.train()  # Ensure model is in training mode
        losses, ppls = [], []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{cfg.num_epochs}', leave=False)
        for X, y in batch_pbar:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, _ = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            ppls.append(perplexity(y_hat, y))
            batch_pbar.set_postfix(loss=f'{loss.item():.4f}', ppl=f'{ppls[-1]:.4f}')
        
        avg_loss = sum(losses) / len(losses)
        avg_ppl = sum(ppls) / len(ppls)
        
        # Evaluate on validation set if provided
        val_ppl, val_loss = None, None
        if val_dataloader is not None:
            val_ppl, val_loss = validate(model, val_dataloader, device=device, loss_fn=loss_fn)
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}', train_ppl=f'{avg_ppl:.4f}', val_ppl=f'{val_ppl:.4f}')
            if cfg.verbose:
                tqdm.write(f'Epoch {epoch + 1}/{cfg.num_epochs} — Loss: {avg_loss:.4f}, '
                           f'Train PPL: {avg_ppl:.4f}, Val PPL: {val_ppl:.4f}')
        else:
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}', ppl=f'{avg_ppl:.4f}')
            if cfg.verbose:
                tqdm.write(f'Epoch {epoch + 1}/{cfg.num_epochs} — Loss: {avg_loss:.4f}, PPL: {avg_ppl:.4f}')
        
        if cfg.logger is not None:
            cfg.logger.log_epoch(epoch, train_loss=avg_loss, train_ppl=avg_ppl, val_ppl=val_ppl, val_loss=val_loss)
    
    if cfg.save_path is not None:
        save_model(model, cfg.save_path)


def validate(model: nn.Module, dataloader: Iterable, device: Optional[torch.device] = None,
             loss_fn: Optional[nn.Module] = None) -> tuple[float, float | None]:
    """Evaluate the model on a validation dataset.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Iterable of validation batches ``(X, y)``.
        device: Torch device; inferred if ``None``.
        loss_fn: Loss function for computing validation loss (optional).

    Returns:
        Tuple of (validation perplexity, validation loss). Loss is None if loss_fn not provided.
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