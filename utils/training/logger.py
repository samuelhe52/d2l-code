"""Training metrics logger (kept separate from stdlib logging)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

History = MutableMapping[str, List[float]]


class TrainingLogger:
    """Logger for tracking training metrics across epochs.
    
    Args:
        log_path: Path to save the log file (default: None, no file saved)
        hparams: Dictionary of hyperparameters to record
    """
    
    def __init__(self, log_path: Optional[str | Path] = None, hparams: Optional[Dict[str, Any]] = None):
        self.log_path = Path(log_path) if log_path else None
        self.hparams: Dict[str, Any] = hparams or {}
        self.start_time = datetime.now()
        self.history: History = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        self.metadata: Dict[str, Any] = {
            'timestamp': self.start_time.isoformat(),
            'hparams': self.hparams,
        }
    
    def log_epoch(self, epoch: int, train_loss: Optional[float] = None, val_loss: Optional[float] = None,
                 train_acc: Optional[float] = None, val_acc: Optional[float] = None, **kwargs: float) -> None:
        """Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss for the epoch
            train_acc: Average training accuracy for the epoch
            val_acc: Validation accuracy after the epoch
            **kwargs: Additional metrics to log
        """
        del epoch  # unused for now; kept for potential future ordering logic
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        
        # Handle additional custom metrics
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def save(self, path: Optional[str | Path] = None) -> None:
        """Save the log to a JSON file, appending to existing runs.
        
        Args:
            path: Path to save (uses log_path from init if not provided)
        """
        save_path = Path(path) if path else self.log_path
        if save_path is None:
            raise ValueError('No log path specified')
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_data: Dict[str, Any] = {
            **self.metadata,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'history': self.history,
        }
        
        # Load existing runs if file exists
        runs: List[Dict[str, Any]] = []
        if save_path.exists():
            with open(save_path, 'r') as f:
                existing = json.load(f)
                # Handle both old format (single run) and new format (list of runs)
                if isinstance(existing, list):
                    runs = existing
                else:
                    runs = [existing]
        
        runs.append(log_data)
        
        with open(save_path, 'w') as f:
            json.dump(runs, f, indent=2)
        print(f'Log saved to {save_path} (run {len(runs)})')
    
    def summary(self) -> None:
        """Print a summary of the training run."""
        duration = datetime.now() - self.start_time
        print(f'\n{"="*50}')
        print('Training Summary')
        print(f'{"="*50}')
        print(f'Duration: {duration}')
        if self.hparams:
            print(f'Hyperparameters: {self.hparams}')
        if self.history['train_loss']:
            print(f'Final train loss: {self.history["train_loss"][-1]:.4f}')
        if self.history['val_loss']:
            print(f'Final val loss:   {self.history["val_loss"][-1]:.4f}')
        if self.history['train_acc']:
            print(f'Final train acc:  {self.history["train_acc"][-1]:.2%}')
        if self.history['val_acc']:
            print(f'Final val acc:    {self.history["val_acc"][-1]:.2%}')
        print(f'{"="*50}\n')
