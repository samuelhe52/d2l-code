"""Base trainer class with common training logic."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from ..io import save_model
from ..training_config import TrainingConfig, resolve_training_config
from ..training_logger import TrainingLogger


def get_device() -> torch.device:
    """Infer the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochMetrics:
    """Container for metrics collected during an epoch."""

    train_loss: float
    val_loss: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_log_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for TrainingLogger.log_epoch."""
        return {"train_loss": self.train_loss, "val_loss": self.val_loss, **self.extra}


class BaseTrainer(ABC):
    """Abstract base class for training PyTorch models.

    Subclasses must implement:
        - ``default_loss_fn``: Property returning the default loss function.
        - ``compute_metrics``: Compute task-specific metrics for a batch.
        - ``format_metrics``: Format metrics for display.
        - ``log_metrics``: Return kwargs for TrainingLogger.log_epoch.

    Optionally override:
        - ``prepare_batch``: Transform batch before forward pass.
        - ``forward``: Customize forward pass (e.g., for RNN state).
        - ``post_backward``: Hook after backward (e.g., gradient clipping).
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable,
        val_dataloader: Iterable | None = None,
        config: TrainingConfig | None = None,
        *,
        num_epochs: int | None = None,
        lr: float | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        save_path: str | None = None,
        verbose: bool = True,
        logger: TrainingLogger | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        # Resolve config with overrides
        self.cfg = resolve_training_config(
            config,
            num_epochs=num_epochs,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            save_path=save_path,
            verbose=verbose,
            logger=logger,
            device=device,
            **kwargs,
        )

        # Setup device
        self.device = self.cfg.device or get_device()
        self.model.to(self.device)

        # Setup loss and optimizer
        self.loss_fn = self.cfg.loss_fn or self.default_loss_fn
        self.optimizer = self.cfg.optimizer or torch.optim.SGD(
            self.model.parameters(), lr=self.cfg.lr
        )

    @property
    @abstractmethod
    def default_loss_fn(self) -> nn.Module:
        """Return the default loss function for this task."""
        ...

    @abstractmethod
    def compute_metrics(self, y_hat: Tensor, y: Tensor, loss: float) -> dict[str, float]:
        """Compute task-specific metrics for a batch.

        Args:
            y_hat: Model predictions.
            y: Ground-truth labels.
            loss: Loss value for this batch.

        Returns:
            Dictionary of metric names to values.
        """
        ...

    @abstractmethod
    def format_train_metrics(self, metrics: dict[str, float]) -> dict[str, str]:
        """Format training metrics for tqdm postfix display.

        Args:
            metrics: Aggregated metrics from compute_metrics.

        Returns:
            Dictionary of formatted strings for display.
        """
        ...

    @abstractmethod
    def format_val_metrics(
        self, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ) -> dict[str, str]:
        """Format combined train/val metrics for tqdm postfix display."""
        ...

    @abstractmethod
    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> dict[str, Any]:
        """Return kwargs for TrainingLogger.log_epoch."""
        ...

    @abstractmethod
    def format_epoch_message(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> str:
        """Format the verbose epoch completion message."""
        ...

    def prepare_batch(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Prepare a batch before forward pass. Override for task-specific transforms."""
        return X.to(self.device), y.to(self.device)

    def forward(self, X: Tensor) -> Tensor:
        """Run forward pass. Override for models with state (e.g., RNNs)."""
        return self.model(X)

    def post_backward(self) -> None:
        """Hook called after loss.backward(). Override for gradient clipping, etc."""
        pass

    def train(self) -> dict[str, float] | None:
        """Run the training loop.

        Returns:
            Final epoch metrics as a dict, or None.
        """
        epoch_pbar = tqdm(range(self.cfg.num_epochs), desc="Training", unit="epoch")
        final_metrics: dict[str, float] | None = None

        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            batch_metrics: list[dict[str, float]] = []

            batch_pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}",
                leave=False,
            )

            for X, y in batch_pbar:
                X, y = self.prepare_batch(X, y)
                self.optimizer.zero_grad()
                y_hat = self.forward(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.post_backward()
                self.optimizer.step()

                metrics = self.compute_metrics(y_hat, y, loss.item())
                batch_metrics.append(metrics)

                # Update batch progress bar
                batch_pbar.set_postfix(self.format_train_metrics(metrics))

            # Aggregate training metrics
            train_metrics = self._aggregate_metrics(batch_metrics)

            # Validation phase
            val_metrics: dict[str, float] | None = None
            if self.val_dataloader is not None:
                val_metrics = self.validate()

            # Update epoch progress bar
            if val_metrics is not None:
                epoch_pbar.set_postfix(self.format_val_metrics(train_metrics, val_metrics))
            else:
                epoch_pbar.set_postfix(self.format_train_metrics(train_metrics))

            # Verbose output
            if self.cfg.verbose:
                msg = self.format_epoch_message(
                    epoch + 1, self.cfg.num_epochs, train_metrics, val_metrics
                )
                tqdm.write(msg)

            # Logging
            if self.cfg.logger is not None:
                log_kwargs = self.log_metrics(epoch, train_metrics, val_metrics)
                self.cfg.logger.log_epoch(epoch, **log_kwargs)

            final_metrics = {**train_metrics, **(val_metrics or {})}

        # Save model
        if self.cfg.save_path is not None:
            save_model(self.model, self.cfg.save_path)

        return final_metrics

    def validate(self) -> dict[str, float]:
        """Run validation and return aggregated metrics."""
        self.model.eval()
        batch_metrics: list[dict[str, float]] = []

        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = self.prepare_batch(X, y)
                y_hat = self.forward(X)
                loss = self.loss_fn(y_hat, y)
                metrics = self.compute_metrics(y_hat, y, loss.item())
                batch_metrics.append(metrics)

        return self._aggregate_metrics(batch_metrics, prefix="val_")

    def _aggregate_metrics(
        self, batch_metrics: list[dict[str, float]], prefix: str = ""
    ) -> dict[str, float]:
        """Average metrics across batches."""
        if not batch_metrics:
            return {}
        keys = batch_metrics[0].keys()
        return {
            f"{prefix}{k}": sum(m[k] for m in batch_metrics) / len(batch_metrics)
            for k in keys
        }
