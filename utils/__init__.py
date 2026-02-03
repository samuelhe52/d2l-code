"""Utility functions for the d2l-code project."""

from typing import List

from . import classfication
from . import regression
from .io import load_model, save_model
from .training_logger import TrainingLogger
from . import rnn

__all__: List[str] = [
    'classfication',
    'regression',
    'save_model',
    'load_model',
    'TrainingLogger',
    'rnn',
    ]