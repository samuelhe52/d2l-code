"""RNN trainer implementation."""

from typing import Any

import torch
from torch import nn, Tensor

from .base import BaseTrainer

class RNNTrainer(BaseTrainer):
    """Trainer for RNN-based models.

    Uses CrossEntropyLoss by default and tracks perplexity as the primary metric.
    Supports gradient clipping via ``grad_clip`` in config.

    The forward pass expects models that return ``(output, state)`` tuples.

    Example:
        >>> config = TrainingConfig(num_epochs=10, lr=0.01, grad_clip=1.0)
        >>> trainer = RNNTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """

    @property
    def default_loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass for RNN models that return (output, state)."""
        y_hat, _ = self.model(X)
        return y_hat

    def post_backward(self) -> None:
        """Apply gradient clipping if configured."""
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

    def compute_metrics(self, y_hat: Tensor, y: Tensor, loss: float) -> dict[str, float]:
        return {
            "loss": loss,
            "ppl": self.perplexity(loss),
        }

    def perplexity(self, loss: float) -> float:
        """Compute perplexity from model outputs and targets."""
        return torch.exp(torch.tensor(loss)).item()

    def format_train_metrics(self, metrics: dict[str, float]) -> dict[str, str]:
        return {
            "loss": f"{metrics['loss']:.4f}",
            "ppl": f"{metrics['ppl']:.4f}",
        }

    def format_val_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> dict[str, str]:
        return {
            "loss": f"{train_metrics['loss']:.4f}",
            "train_ppl": f"{train_metrics['ppl']:.4f}",
            "val_ppl": f"{val_metrics['val_ppl']:.4f}",
        }

    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "train_loss": train_metrics["loss"],
            "train_ppl": train_metrics["ppl"],
        }
        if val_metrics:
            result["val_loss"] = val_metrics.get("val_loss")
            result["val_ppl"] = val_metrics.get("val_ppl")
        return result

    def format_epoch_message(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> str:
        base = f"Epoch {epoch}/{num_epochs} — Loss: {train_metrics['loss']:.4f}"
        if val_metrics:
            return f"{base}, Train PPL: {train_metrics['ppl']:.4f}, Val PPL: {val_metrics['val_ppl']:.4f}"
        return f"{base}, PPL: {train_metrics['ppl']:.4f}"
