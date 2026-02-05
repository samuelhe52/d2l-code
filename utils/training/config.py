"""Shared training configuration utilities."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch.optim import Optimizer

if TYPE_CHECKING:
    from .logger import TrainingLogger


@dataclass
class TrainingConfig:
    """Container for common training hyperparameters.

    Explicit function arguments will override these values when both are set.
    """

    num_epochs: Optional[int] = None
    lr: Optional[float] = None
    loss_fn: Optional[Callable[..., torch.Tensor]] = None
    optimizer: Optional[Optimizer] = None
    save_path: Optional[str] = None
    verbose: bool = True
    logger: Optional["TrainingLogger"] = None
    device: Optional[torch.device] = None
    grad_clip: Optional[float] = None  # Max norm for gradient clipping


def resolve_training_config(config: Optional[TrainingConfig], **overrides: Any) -> TrainingConfig:
    """Merge a ``TrainingConfig`` with explicit overrides.

    Args:
        config: Base configuration (optional).
        **overrides: Values passed directly to the training function.

    Returns:
        A ``TrainingConfig`` with explicit overrides taking precedence.
    """
    base = config or TrainingConfig()
    merged = {name: getattr(base, name) for name in TrainingConfig.__dataclass_fields__}
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    resolved = TrainingConfig(**merged)
    if resolved.num_epochs is None:
        raise ValueError("num_epochs must be provided either directly or via TrainingConfig")
    if resolved.lr is None:
        raise ValueError("lr must be provided either directly or via TrainingConfig")
    return resolved
