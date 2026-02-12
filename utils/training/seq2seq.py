import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .rnn import RNNTrainer
from typing import Tuple

def masked_ce_loss(y_hat: Tensor, y: Tuple[Tensor, int]) -> Tensor:
    """
    Compute the masked cross-entropy loss for sequence-to-sequence models.
    
    Args:
        y_hat: Predicted outputs (batch_size, vocab_size, seq_len)
        y: Tuple containing:
            - Actual target sequences (batch_size, seq_len)
            - Padding token index (int)
    """
    tgt_outputs, pad_token_index = y
    tgt_outputs = tgt_outputs.to(torch.long)
    
    return F.cross_entropy(y_hat, tgt_outputs,
                           ignore_index=pad_token_index, reduction='mean')
        

class Seq2SeqTrainer(RNNTrainer):
    """
    Trainer for Sequence-to-Sequence models.
    
    Requires a Seq2Seq model and uses masked cross-entropy loss.
    """
    @property
    def default_loss_fn(self):
        return masked_ce_loss

    def prepare_batch(self, X, y):
        src, tgt_array, src_valid_len = X
        src = src.to(self.device)
        tgt_array = tgt_array.to(self.device)
        src_valid_len = src_valid_len.to(self.device)
        return ((src, tgt_array, src_valid_len),
                (y.to(self.device), self.model.pad_token_index))
    
    def forward(self, X: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        X_src, X_tgt, src_valid_len = X
        return self.model(X_src, X_tgt, src_valid_len)
