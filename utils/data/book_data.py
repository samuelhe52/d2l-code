"""Book text data utilities for RNN/language model training."""

import hashlib
import re
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .vocab import Vocab


class BookData(Dataset):
    """Dataset for loading and processing book text data.

    Args:
        seq_len: Length of each sequence sample.
        book_name: Name of the book.
        book_url: URL to download the book text.
        md5_hash: MD5 hash to verify the downloaded file.
        data_root: Root directory for storing/loading the dataset.
        use_chars: Whether to tokenize the text into characters (True) or words (False).
        vocab: Optional vocabulary to reuse for consistent token indices across books.
    """

    def __init__(
        self,
        seq_len: int = 35,
        *,
        book_name: str,
        book_url: str,
        md5_hash: str,
        data_root: str = "./data",
        use_chars: bool = True,
        vocab: Vocab | None = None,
    ):
        super().__init__()
        self.book_name = book_name
        self.book_url = book_url
        self.md5_hash = md5_hash
        self.data_root = data_root

        self.text = self._load_data_str(data_root)
        self.processed_text = self._preprocess_text(self.text)
        self.tokens = self._tokenize(self.processed_text, use_chars=use_chars)
        # Reuse a supplied vocabulary to align with a previously trained model
        self.vocab = vocab if vocab is not None else Vocab(self.tokens)
        self.corpus = [self.vocab[token] for token in self.tokens]

        self.features, self.labels = self._create_features_and_labels(seq_len)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _load_data_str(self, data_root) -> str:
        """Fetch the book text data as a string."""
        url = self.book_url
        file_name = f"{self.book_name}.txt"
        file_path = data_root + "/" + file_name
        if not Path(file_path).exists() or not self._check_md5(
            file_path, self.md5_hash
        ):
            torch.hub.download_url_to_file(url, file_path)
        with open(file_path, "r") as f:
            text = f.read()
        return text

    def _check_md5(self, file_path: str, md5_hash: str) -> bool:
        """Check the MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest() == md5_hash

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text. Subclasses should implement this method."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _tokenize(self, text: str, use_chars: bool) -> list[str]:
        """Tokenize the text into characters or words."""
        return list(text) if use_chars else text.split()

    def _create_features_and_labels(self, seq_len: int) -> tuple[Tensor, Tensor]:
        """Create features and labels tensors for sequence modeling."""
        num_examples = len(self.corpus) - seq_len
        array = torch.tensor(
            [self.corpus[i : i + seq_len + 1] for i in range(num_examples)]
        )
        features = array[:, :-1]
        labels = array[:, 1:]
        return features, labels


class TimeMachineData(BookData):
    """Dataset for "The Time Machine" by H.G. Wells.

    Args:
        seq_len: Length of each sequence sample.
        data_root: Root directory for storing/loading the dataset.
        use_chars: Whether to tokenize into characters (True) or words (False).
    """

    def __init__(
        self, seq_len: int, data_root: str = "./data", use_chars: bool = True
    ):
        super().__init__(
            seq_len=seq_len,
            book_name="time_machine",
            book_url="https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt",
            md5_hash="7353d136ab308ecd0d4f1c5bf0e23122",
            data_root=data_root,
            use_chars=use_chars,
        )

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.strip()


class PrideAndPrejudiceData(BookData):
    """Dataset for "Pride and Prejudice" by Jane Austen.

    Args:
        seq_len: Length of each sequence sample.
        data_root: Root directory for storing/loading the dataset.
        use_chars: Whether to tokenize into characters (True) or words (False).
        vocab: Optional vocabulary to reuse for consistent token indices.
    """

    def __init__(
        self,
        seq_len: int,
        data_root: str = "./data",
        use_chars: bool = True,
        vocab: Vocab | None = None,
    ):
        super().__init__(
            seq_len=seq_len,
            book_name="pride_and_prejudice",
            book_url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            md5_hash="9ec834c0167fbb97231ffa192f75b09a",
            data_root=data_root,
            use_chars=use_chars,
            vocab=vocab,
        )

    def _preprocess_text(self, text: str) -> str:
        header_pattern = r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*\n"
        footer_pattern = r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*"
        start = re.search(header_pattern, text)
        end = re.search(footer_pattern, text)
        if start and end:
            text = text[start.end() : end.start()]

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.strip()


class WarOfTheWorldsData(BookData):
    """Dataset for "The War of the Worlds" by H.G. Wells.

    Args:
        seq_len: Length of each sequence sample.
        data_root: Root directory for storing/loading the dataset.
        use_chars: Whether to tokenize into characters (True) or words (False).
        vocab: Optional vocabulary to reuse for consistent token indices.
    """

    def __init__(
        self,
        seq_len: int,
        data_root: str = "./data",
        use_chars: bool = True,
        vocab: Vocab | None = None,
    ):
        super().__init__(
            seq_len=seq_len,
            book_name="war_of_the_worlds",
            book_url="https://www.gutenberg.org/cache/epub/36/pg36.txt",
            md5_hash="5d0de2070465618da7d621e309e1a164",
            data_root=data_root,
            use_chars=use_chars,
            vocab=vocab,
        )

    def _preprocess_text(self, text: str) -> str:
        header_pattern = r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*\n"
        footer_pattern = r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*"
        start = re.search(header_pattern, text)
        end = re.search(footer_pattern, text)
        if start and end:
            text = text[start.end() : end.start()]

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.strip()


def book_data_loader(
    book_data: BookData, batch_size: int, train_ratio: float = 0.8, train: bool = True
) -> DataLoader:
    """Create a DataLoader for the given BookData dataset.

    Args:
        book_data: A BookData dataset instance.
        batch_size: Batch size for the DataLoader.
        train_ratio: Ratio of data to use for training.
        train: Whether to return the training DataLoader or validation DataLoader.

    Returns:
        DataLoader yielding (features, labels).
    """
    n_train = int(len(book_data) * train_ratio)
    if train:
        subset = torch.utils.data.Subset(book_data, list(range(n_train)))
    else:
        subset = torch.utils.data.Subset(
            book_data, list(range(n_train, len(book_data)))
        )
    return DataLoader(subset, batch_size=batch_size, shuffle=train)
