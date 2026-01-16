"""Training utilities for regression tasks."""

import torch
from torch import nn
from tqdm import tqdm

from ..io import save_model

def train(model, dataloader, num_epochs, lr,
          loss_fn=None, optimizer=None, save_path=None, 
          verbose=True, logger=None, val_dataloader=None,
          device=None):
    """Train a regression model.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        num_epochs: Number of epochs to train
        lr: Learning rate
        loss_fn: Loss function (default: MSELoss)
        optimizer: Optimizer (default: SGD with specified lr)
        save_path: Path to save model parameters after training (default: None)
        verbose: Whether to print loss after each epoch (default: True)
        logger: TrainingLogger instance for logging metrics (default: None)
        val_dataloader: DataLoader for validation data to evaluate after each epoch (default: None)
        device: Torch device to run training on, e.g. torch.device('cuda') (default: None, infers from model or uses CPU)

    Returns:
        A tuple for average (training loss, validation loss) for the last epoch.
        If no validation dataloader is provided, validation loss will be None.
    """
    # Infer device if not explicitly provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.to(device)

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
    last_train_loss = None
    last_val_loss = None
    for epoch in epoch_pbar:
        model.train() # Ensure model is in training mode
        losses = []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
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
            if verbose:
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}, '
                        f'Val Loss: {val_loss:.4f}')
        else:
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}')
            if verbose:
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}')
        
        if logger is not None:
            logger.log_epoch(epoch, train_loss=avg_loss, val_loss=val_loss)
    
    if save_path is not None:
        save_model(model, save_path)

    return last_train_loss, last_val_loss


def validate(model, loss_fn, dataloader, device=None):
    """Evaluate the model on a validation dataset.
    
    Args:
        model: PyTorch model to evaluate
        loss_fn: Loss function to use for evaluation
        dataloader: DataLoader for validation data
        device: Torch device to run evaluation on (default: None, infers from model or uses CPU)
        
    Returns:
        Average loss over the validation dataset.
    """
    # Infer device if not explicitly provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
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
