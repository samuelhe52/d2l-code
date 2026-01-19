"""Training utilities for regression tasks."""

import torch
from torch import nn
from tqdm import tqdm

from ..io import save_model
from ..training_config import TrainingConfig, resolve_training_config

def train(model, dataloader, num_epochs=None, lr=None,
          loss_fn=None, optimizer=None, save_path=None, 
          verbose=True, logger=None, val_dataloader=None,
          device=None, config: TrainingConfig | None = None):
    """Train a regression model.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        num_epochs: Number of epochs to train. Can be provided via ``config``.
        lr: Learning rate. Can be provided via ``config``.
        loss_fn: Loss function (default: MSELoss)
        optimizer: Optimizer (default: SGD with specified lr)
        save_path: Path to save model parameters after training (default: None)
        verbose: Whether to print loss after each epoch (default: True)
        logger: TrainingLogger instance for logging metrics (default: None)
        val_dataloader: DataLoader for validation data to evaluate after each epoch (default: None)
        device: Torch device to run training on, e.g. torch.device('cuda')
            (default: None, tries cuda, mps, cpu)
        config: Optional ``TrainingConfig``. Explicit args override matching fields.

    Returns:
        A tuple for average (training loss, validation loss) for the last epoch.
        If no validation dataloader is provided, validation loss will be None.
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
    last_train_loss = None
    last_val_loss = None
    for epoch in epoch_pbar:
        model.train() # Ensure model is in training mode
        losses = []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{cfg.num_epochs}', leave=False)
        for X, y in batch_pbar:
            X = X.to(device)
            y = y.to(device)
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


def validate(model, loss_fn, dataloader, device=None):
    """Evaluate the model on a validation dataset.
    
    Args:
        model: PyTorch model to evaluate
        loss_fn: Loss function to use for evaluation
        dataloader: DataLoader for validation data
        device: Torch device to run evaluation on (default: None, tries cuda, mps, cpu)
        
    Returns:
        Average loss over the validation dataset.
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
            y = y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return avg_loss
