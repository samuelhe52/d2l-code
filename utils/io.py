"""Model I/O utilities."""

from pathlib import Path
from typing import Union

import torch
from torch.nn import Module

PathLike = Union[str, Path]


def save_model(model: Module, save_path: PathLike) -> None:
    """Save model parameters to a file.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save model parameters
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


def load_model(load_path: PathLike, model: Module) -> Module:
    """Load model parameters from a file.
    
    Args:
        load_path: Path to the saved model parameters
        model: PyTorch model to load parameters into
        
    Returns:
        The model with loaded parameters
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f'No model found at {load_path}')
    state_dict = torch.load(load_path, weights_only=True)
    model.load_state_dict(state_dict)
    print(f'Model loaded from {load_path}')
    return model
