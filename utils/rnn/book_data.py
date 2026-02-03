import random
import re
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import hashlib
from .vocab import Vocab

class BookData(Dataset):
    """
    Dataset for loading and processing book text data.
    
    Args:
        book_name (str): Name of the book.
        book_url (str): URL to download the book text.
        md5_hash (str): MD5 hash to verify the downloaded file.
        data_root (str): Root directory for storing/loading the dataset.
        use_chars (bool): Whether to tokenize the text into 
            characters (True) or words (False).
    """
    def __init__(self, book_name: str, book_url: str, md5_hash: str,
                 data_root: str = './data', use_chars: bool = True):
        super().__init__()
        self.book_name = book_name
        self.book_url = book_url
        self.md5_hash = md5_hash
        self.data_root = data_root
        
        self.text = self._load_data_str(data_root)
        self.processed_text = self._preprocess_text(self.text)
        self.tokens = self._tokenize(self.processed_text, use_chars=use_chars)
        self.vocab = Vocab(self.tokens)
        self.corpus = [self.vocab[token] for token in self.tokens]

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
    
class TimeMachineData(BookData):
    """
    Dataset for the "The Time Machine" book by H.G. Wells.
    """
    def __init__(self, data_root: str = './data', use_chars: bool = True):
        super().__init__(
            book_name='time_machine',
            book_url='https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt',
            md5_hash='7353d136ab308ecd0d4f1c5bf0e23122',
            data_root=data_root,
            use_chars=use_chars
        )
    
    def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters (keep spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.strip()


class PrideAndPrejudiceData(BookData):
    """Dataset for Jane Austen's "Pride and Prejudice."""

    def __init__(self, data_root: str = './data', use_chars: bool = True):
        super().__init__(
            book_name='pride_and_prejudice',
            book_url='https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
            md5_hash='9ec834c0167fbb97231ffa192f75b09a',
            data_root=data_root,
            use_chars=use_chars,
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
