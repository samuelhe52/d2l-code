"""Book data utilities.

.. deprecated::
    This module is kept for backward compatibility.
    Use ``utils.data`` instead for:
    - ``BookData``
    - ``TimeMachineData``
    - ``PrideAndPrejudiceData``
    - ``WarOfTheWorldsData``
    - ``book_data_loader``
"""

from ..data.book_data import (
    BookData,
    TimeMachineData,
    PrideAndPrejudiceData,
    WarOfTheWorldsData,
    book_data_loader,
)
from ..data.vocab import Vocab

__all__ = [
    "BookData",
    "TimeMachineData",
    "PrideAndPrejudiceData",
    "WarOfTheWorldsData",
    "book_data_loader",
    "Vocab",
]
