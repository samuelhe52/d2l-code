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


def train(model, dataloader, num_epochs, lr=0.01,
          loss_fn=None, optimizer=None, save_path=None):
    """Train a classification model.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        num_epochs: Number of epochs to train
        lr: Learning rate (default: 0.01)
        loss_fn: Loss function (default: CrossEntropyLoss)
        optimizer: Optimizer (default: SGD with specified lr)
        save_path: Path to save model parameters after training (default: None)
    """
    model.train()
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        losses, accuracies = [], []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for X, y in pbar:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append(accuracy(y_hat, y))
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{accuracies[-1]:.4f}')
        print(f'Epoch {epoch + 1}, '
              f'Loss: {sum(losses) / len(losses):.4f}, '
              f'Accuracy: {sum(accuracies) / len(accuracies):.4f}')
    
    if save_path is not None:
        save_model(model, save_path)


def test(model, dataloader):
    """Evaluate the model on a test dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for test data
        
    Returns:
        Test accuracy as a float between 0 and 1
    """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            accuracies.append(accuracy(y_hat, y))
    return sum(accuracies) / len(accuracies)
