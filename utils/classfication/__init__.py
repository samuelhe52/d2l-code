"""Classification utilities."""

from .training import train, validate, accuracy
from .data import fashion_mnist

__all__ = ['train', 'validate', 'accuracy', 'fashion_mnist']