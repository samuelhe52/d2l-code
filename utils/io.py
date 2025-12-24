"""Model I/O utilities."""

from pathlib import Path

import torch


def save_model(model, save_path):
    """Save model parameters to a file.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save model parameters
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


def load_model(model, load_path):
    """Load model parameters from a file.
    
    Args:
        model: PyTorch model to load parameters into
        load_path: Path to the saved model parameters
        
    Returns:
        The model with loaded parameters
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f'No model found at {load_path}')
    model.load_state_dict(torch.load(load_path, weights_only=True))
    print(f'Model loaded from {load_path}')
    return model
