import re
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import hashlib
from .vocab import Vocab

class BookData(Dataset):
    """
    Dataset for loading and processing book text data.
    
    Args:
        seq_len (int): Length of each sequence sample.
        book_name (str): Name of the book.
        book_url (str): URL to download the book text.
        md5_hash (str): MD5 hash to verify the downloaded file.
        data_root (str): Root directory for storing/loading the dataset.
        use_chars (bool): Whether to tokenize the text into 
            characters (True) or words (False).
    """
    def __init__(self, seq_len: int = 35, *, book_name: str,
                 book_url: str, md5_hash: str, data_root: str = './data',
                 use_chars: bool = True, vocab: Vocab | None = None):
        super().__init__()
        self.book_name = book_name
        self.book_url = book_url
        self.md5_hash = md5_hash
        self.data_root = data_root
        
        self.text = self._load_data_str(data_root)
        self.processed_text = self._preprocess_text(self.text)
        self.tokens = self._tokenize(self.processed_text, use_chars=use_chars)
        # Reuse a supplied vocabulary to align with a previously trained model
        # (e.g., evaluating on a new book with the Time Machine vocabulary).
        self.vocab = vocab if vocab is not None else Vocab(self.tokens)
        self.corpus = [self.vocab[token] for token in self.tokens]

        self.features, self.labels = self._create_features_and_labels(seq_len)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _load_data_str(self, data_root) -> str:
        """
        Fetch the book text data as a string.
        
        Args:
            data_root (str): Root directory for storing/loading the dataset.
        """
        url = self.book_url
        file_name = f'{self.book_name}.txt'
        file_path = data_root + '/' + file_name
        # This will download the file if it does not exist
        # If the file already exists, it will not download it again
        if not Path(file_path).exists() or not \
            self._check_md5(file_path, self.md5_hash):
                torch.hub.download_url_to_file(url, file_path)
        with open(file_path, 'r') as f:
            text = f.read()
        return text
    
    def _check_md5(self, file_path: str, md5_hash: str) -> bool:
        """
        Check the MD5 hash of a file.
        
        Args:
            file_path (str): Path to the file.
            md5_hash (str): Expected MD5 hash.
            
        Returns:
            bool: True if the file's MD5 hash matches the expected hash, False otherwise.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest() == md5_hash

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by converting to lowercase and removing non-alphabetic characters.
        
        Args:
            text (str): Raw text data.
            
        Returns:
            str: Preprocessed text data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _tokenize(self, text: str, use_chars) -> list[str]:
        """
        Tokenize the text into characters or words.
        
        Args:
            text (str): Preprocessed text data.
            use_chars (bool): Whether to tokenize into characters (True) or words (False).
        """
        return list(text) if use_chars else text.split()
    
    def _create_features_and_labels(self, seq_len: int) -> tuple[Tensor, Tensor]:
        """
        Create features and labels tensors for sequence modeling.
        
        Args:
            seq_len (int): Length of each sequence sample.
        Returns:
            tuple[Tensor, Tensor]: Features and labels tensors.
        """
        num_examples = len(self.corpus) - seq_len
        array = torch.tensor(
            [self.corpus[i : i + seq_len + 1]
             for i in range(num_examples)]
        )
        features = array[:, :-1]
        labels = array[:, 1:]
        return features, labels
    
class TimeMachineData(BookData):
    """
    Dataset for the "The Time Machine" book by H.G. Wells.
    
    Args:
        seq_len (int): Length of each sequence sample.
        data_root (str): Root directory for storing/loading the dataset.
        use_chars (bool): Whether to tokenize the text into 
            characters (True) or words (False).
    """
    def __init__(self, seq_len: int, data_root: str = './data', use_chars: bool = True):
        super().__init__(
            seq_len=seq_len,
            book_name='time_machine',
            book_url='https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt',
            md5_hash='7353d136ab308ecd0d4f1c5bf0e23122',
            data_root=data_root,
            use_chars=use_chars,
        )
    
    def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters (keep spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.strip()


class PrideAndPrejudiceData(BookData):
    """
    Dataset for Jane Austen's "Pride and Prejudice.
    
    Args:
        seq_len (int): Length of each sequence sample.
        data_root (str): Root directory for storing/loading the dataset.
        use_chars (bool): Whether to tokenize the text into 
            characters (True) or words (False).
    """
    def __init__(self, seq_len: int, data_root: str = './data',
                 use_chars: bool = True, vocab: Vocab | None = None):
        super().__init__(
            seq_len=seq_len,
            book_name='pride_and_prejudice',
            book_url='https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
            md5_hash='9ec834c0167fbb97231ffa192f75b09a',
            data_root=data_root,
            use_chars=use_chars,
            vocab=vocab,
        )

    def _preprocess_text(self, text: str) -> str:
        # Strip Gutenberg header/footer when present to keep only the novel body.
        header_pattern = r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*\n"
        footer_pattern = r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*"
        start = re.search(header_pattern, text)
        end = re.search(footer_pattern, text)
        if start and end:
            text = text[start.end():end.start()]

        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.strip()


class WarOfTheWorldsData(BookData):
    """
    Dataset for H.G. Wells' "The War of the Worlds".

    Args:
        seq_len (int): Length of each sequence sample.
        data_root (str): Root directory for storing/loading the dataset.
        use_chars (bool): Whether to tokenize the text into 
            characters (True) or words (False).
        vocab (Vocab | None): Optional vocabulary to reuse for consistent
            token indices across books.
    """
    def __init__(self, seq_len: int, data_root: str = './data',
                 use_chars: bool = True, vocab: Vocab | None = None):
        super().__init__(
            seq_len=seq_len,
            book_name='war_of_the_worlds',
            book_url='https://www.gutenberg.org/cache/epub/36/pg36.txt',
            md5_hash='5d0de2070465618da7d621e309e1a164',
            data_root=data_root,
            use_chars=use_chars,
            vocab=vocab,
        )

    def _preprocess_text(self, text: str) -> str:
        # Strip Gutenberg boilerplate when present to keep only the novel body.
        header_pattern = r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*\n"
        footer_pattern = r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*"
        start = re.search(header_pattern, text)
        end = re.search(footer_pattern, text)
        if start and end:
            text = text[start.end():end.start()]

        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.strip()

def book_data_loader(book_data: BookData, batch_size: int,
                     train_ratio: float = 0.8, train: bool = True) -> DataLoader:
    """
    Create a DataLoader for the given BookData dataset.
    
    Args:
        book_data (BookData): A BookData dataset instance.
        batch_size (int): Batch size for the DataLoader.
        train_ratio (float): Ratio of data to use for training.
        train (bool): Whether to return the training DataLoader or validation DataLoader.
    Returns:
        DataLoader: Torch DataLoader yielding (features, labels).
    """
    n_train = int(len(book_data) * train_ratio)
    if train:
        subset = torch.utils.data.Subset(book_data, list(range(n_train)))
    else:
        subset = torch.utils.data.Subset(book_data, list(range(n_train, len(book_data))))
    return DataLoader(subset, batch_size=batch_size, shuffle=train)
