import collections
from torch.utils.data import Dataset

class Vocab(Dataset):
    """
    Vocabulary for mapping characters to indices and vice versa.
    
    Args:
        tokens (list[str]): List of tokens (characters).
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.
        reserved_tokens (list[str]): List of reserved tokens (e.g., padding, unknown).
    """
    def __init__(self, tokens: list[str] = [],
                 min_freq: int = 0, reserved_tokens: list[str] = []):
        # Flatten if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        # collections.Counter() returns a dict subclass containing token counts
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        unique_tokens = set(['unk'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq
        ])
        # Sort tokens, use numerical order to index into the vocabulary
        self.idx_to_token = sorted(list(unique_tokens))
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        
    def __len__(self) -> int:
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            # Get the index of a single token, return 'unk' index if not found
            return self.token_to_idx.get(tokens, self.token_to_idx['unk'])
        elif isinstance(tokens, list):
            return [self.__getitem__(token) for token in tokens]
        else:
            raise TypeError('Input must be a string or list of strings')
        
    def to_tokens(self, indices: int | list[int]) -> str | list[str]:
        if isinstance(indices, int):
            return self.idx_to_token[indices]
        elif isinstance(indices, list):
            return [self.idx_to_token[index] for index in indices]
        else:
            raise TypeError('Input must be an integer or list of integers')
        
    @property
    def unk(self) -> int:
        """Index of the unknown token."""
        return self.token_to_idx['unk']