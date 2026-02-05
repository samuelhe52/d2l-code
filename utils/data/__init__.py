"""Unified data loading utilities.

This module provides data loading utilities for all task types:
- Classification: FashionMNIST
- RNN/Language Models: BookData, Vocab, and specific book datasets
"""

from .classification import fashion_mnist
from .vocab import Vocab
from .book_data import (
    BookData,
    TimeMachineData,
    PrideAndPrejudiceData,
    WarOfTheWorldsData,
    book_data_loader,
)

__all__ = [
    # Classification
    "fashion_mnist",
    # Vocabulary
    "Vocab",
    # Book/RNN data
    "BookData",
    "TimeMachineData",
    "PrideAndPrejudiceData",
    "WarOfTheWorldsData",
    "book_data_loader",
]
