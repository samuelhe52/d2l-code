"""Training utilities for classification tasks."""

import torch
from torch import nn
from tqdm import tqdm

from ..io import save_model


def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    
    Args:
        y_hat: Predicted logits, shape (batch_size, num_classes)
        y: True labels, shape (batch_size,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    preds = y_hat.argmax(dim=1)
    return (preds == y).type(torch.float).sum().item() / len(y)

def train(model, dataloader, num_epochs, lr,
          loss_fn=None, optimizer=None, save_path=None, 
          verbose=True, logger=None, val_dataloader=None):
    """Train a classification model.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        num_epochs: Number of epochs to train
        lr: Learning rate (default: 0.01)
        loss_fn: Loss function (default: CrossEntropyLoss)
        optimizer: Optimizer (default: SGD with specified lr)
        save_path: Path to save model parameters after training (default: None)
        verbose: Whether to print loss after each epoch (default: True)
        logger: TrainingLogger instance for logging metrics (default: None)
        val_dataloader: DataLoader for validation data to evaluate after each epoch (default: None)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
    for epoch in epoch_pbar:
        model.train() # Ensure model is in training mode
        losses, accuracies = [], []
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for X, y in batch_pbar:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append(accuracy(y_hat, y))
            batch_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{accuracies[-1]:.4f}')
        
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accuracies) / len(accuracies)
        
        # Evaluate on validation set if provided
        val_acc = None
        if val_dataloader is not None:
            val_acc = validate(model, val_dataloader)
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}', train=f'{avg_acc:.2%}', val=f'{val_acc:.2%}')
            if verbose:
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}, '
                           f'Train: {avg_acc:.2%}, Val: {val_acc:.2%}')
        else:
            epoch_pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.2%}')
            if verbose:
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}')
        
        if logger is not None:
            logger.log_epoch(epoch, train_loss=avg_loss, train_acc=avg_acc, val_acc=val_acc)
    
    if save_path is not None:
        save_model(model, save_path)


def validate(model, dataloader):
    """Evaluate the model on a validation dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for validation data
        
    Returns:
        Validation accuracy as a float between 0 and 1
    """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            accuracies.append(accuracy(y_hat, y))
    return sum(accuracies) / len(accuracies)
