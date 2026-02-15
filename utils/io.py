"""Model I/O utilities."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from torch.nn import Module

PathLike = Union[str, Path]


def _resolve_save_path(model: Module, save_path: PathLike) -> Path:
    """Resolve a model save path.

    If ``save_path`` is a directory or has no suffix, generate a unique filename
    based on the model class and timestamp.
    """
    path = Path(save_path)
    if path.suffix:
        return path

    dir_path = path
    dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    base_name = f"{model_name}_{timestamp}"
    candidate = dir_path / f"{base_name}.pt"
    counter = 1
    while candidate.exists():
        candidate = dir_path / f"{base_name}_{counter}.pt"
        counter += 1
    return candidate


def save_model(model: Module, save_path: PathLike) -> None:
    """Save model parameters to a file.

    Args:
        model: PyTorch model to save
        save_path: File path or directory to save model parameters
    """
    resolved_path = _resolve_save_path(model, save_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), resolved_path)
    print(f'Model saved to {resolved_path}')


def _resolve_latest_model_path(load_path: Path) -> Path:
    """Resolve the most recent model checkpoint from a directory."""
    candidates = [p for p in load_path.iterdir() if p.is_file() and p.suffix == ".pt"]
    if not candidates:
        raise FileNotFoundError(f'No model checkpoints found in {load_path}')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(load_path: PathLike, model: Module,
               device: Optional[torch.device] = None) -> Module:
    """Load model parameters from a file.

    Args:
        load_path: Path to the saved model parameters (file or directory)
        model: PyTorch model to load parameters into
        device: Device on which to map the loaded parameters
    Returns:
        The model with loaded parameters
    """
    if device is None:
        from .training.base import get_device
        device = get_device()
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f'No model found at {load_path}')
    if load_path.is_dir():
        load_path = _resolve_latest_model_path(load_path)
    state_dict = torch.load(load_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    print(f'Model loaded from {load_path}')
    return model
