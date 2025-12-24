"""Training utilities for regression tasks."""

import torch
from torch import nn
from tqdm import tqdm

from ..io import save_model

def train(model, dataloader, num_epochs, lr,
          loss_fn=None, optimizer=None, save_path=None, 
          logger=None, test_dataloader=None):
    """Train a regression model.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        num_epochs: Number of epochs to train
        lr: Learning rate
        loss_fn: Loss function (default: MSELoss)
        optimizer: Optimizer (default: SGD with specified lr)
        save_path: Path to save model parameters after training (default: None)
        logger: TrainingLogger instance for logging metrics (default: None)
        test_dataloader: DataLoader for test data to evaluate after each epoch (default: None)
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
    for epoch in epoch_pbar:
        model.train() # Ensure model is in training mode
        losses = []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for X, y in batch_pbar:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            batch_pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = sum(losses) / len(losses)
        
        # Evaluate on test set if provided
        test_loss = None
        if test_dataloader is not None:
            test_loss = test(model, loss_fn, test_dataloader)
            epoch_pbar.set_postfix(train=f'{avg_loss:.4f}', test=f'{test_loss:.4f}')
            tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}, '
                       f'Test Loss: {test_loss:.4f}')
        else:
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}')
            tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}')
        
        if logger is not None:
            logger.log_epoch(epoch, train_loss=avg_loss, test_loss=test_loss)
    
    if save_path is not None:
        save_model(model, save_path)


def test(model, loss_fn, dataloader):
    """Evaluate the model on a test dataset.
    
    Args:
        model: PyTorch model to evaluate
        loss_fn: Loss function to use for evaluation
        dataloader: DataLoader for test data
        
    Returns:
        Average loss over the test dataset.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return avg_loss
