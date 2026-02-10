import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.enc_dec import Encoder, Decoder, EncoderDecoder
from utils.training import RNNTrainer, TrainingConfig, TrainingLogger
from utils.data.mt_data import (
    FraEngDataset,
    GerEngDataset,
    mt_dataloader,
    eval_translations,
)
from utils.io import load_model
from typing import Optional, Tuple

class Seq2SeqEncoder(Encoder):
    """
    Sequence-to-Sequence Encoder for Machine Translation, implemented using GRU.
    """
    def __init__(self, vocab_size: int, embed_size: int,
                 num_hiddens: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout, batch_first=False)
        
    def forward(self, X: Tensor, *args) -> Tuple[Tensor, Tensor]:
        # X shape: (batch_size, seq_len)
        embs = self.embedding(X.T.to(torch.long))
        # embs shape: (seq_len, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (seq_len, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state


class Seq2SeqDecoder(Decoder):
    """
    Sequence-to-Sequence Decoder for Machine Translation, implemented using GRU.
    """
    def __init__(self, vocab_size: int, embed_size: int,
                 num_hiddens: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Concatenate the context vector and the input embedding
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout, batch_first=False)
        # Dense layer to convert decoder output vectors to vocabulary space
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def preprocess_state(self, enc_outputs, *args):
        outputs, hidden_state = enc_outputs
        src_valid_len = args[0] if len(args) > 0 else None
        return outputs, hidden_state, src_valid_len
    
    def forward(self, X: Tensor, state: Tuple[Tensor, Tensor, Tensor | None]) \
        -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor | None]]:
        embs = self.embedding(X.T.to(torch.long))
        # embs shape: (seq_len, batch_size, embed_size)
        # enc_outputs shape: (seq_len, batch_size, num_hiddens)
        enc_outputs, hidden_state, src_valid_len = state
        # src_valid_len shape: (batch_size,)
        if src_valid_len is None:
            # Use -1 index to get the last output for all sequences
            context = enc_outputs[-1] # Shape: (batch_size, num_hiddens)
        else:
            lengths = src_valid_len.to(torch.long).clamp_min(1) - 1
            batch_idx = torch.arange(
                enc_outputs.shape[1], device=enc_outputs.device
            )
            # Advanced indexing to get the last valid output for each sequence
            context = enc_outputs[lengths, batch_idx]
        # Shape: (batch_size, num_hiddens)
        # Repeat the context vector for each time step
        # Original context shape: (batch_size, num_hiddens)
        # Repeated context shape: (seq_len, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concatenate the context vector with the input embeddings
        rnn_inputs = torch.cat((embs, context), dim=-1)
        outputs, hidden_state = self.rnn(rnn_inputs, hidden_state)
        # Raw outputs shape: (seq_len, batch_size, num_hiddens)
        outputs = self.dense(outputs).permute(1, 2, 0)
        # outputs shape: (batch_size, vocab_size, seq_len)
        return outputs, (enc_outputs, hidden_state, src_valid_len)
    

class Seq2Seq(EncoderDecoder):
    """
    Sequence-to-Sequence model for Machine Translation.
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder, 
                 decoder: Seq2SeqDecoder,
                 pad_token_index: int):
        super().__init__(encoder, decoder)
        self.pad_token_index = pad_token_index
        

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
    
    
if __name__ == "__main__":
    hparams = {
        'seq_len': 25,
        'batch_size': 128,
        'num_epochs': 15,
        'lr': 2e-3,
        'grad_clip': 1.0,
        'embed_size': 256,
        'num_hiddens': 512,
        'num_layers': 2,
        'dropout': 0.2,
    }
    
    data = GerEngDataset(
        seq_len=hparams['seq_len'],
        token_min_freq=5,
        # total_samples=20000,
    )
    train_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=False
    )
    
    shared = {
        'embed_size': hparams['embed_size'],
        'num_hiddens': hparams['num_hiddens'],
        'num_layers': hparams['num_layers'],
        'dropout': hparams['dropout'],
    }
    
    encoder = Seq2SeqEncoder(vocab_size=len(data.src_vocab), **shared)
    decoder = Seq2SeqDecoder(vocab_size=len(data.tgt_vocab), **shared)
    model = Seq2Seq(
        encoder=encoder, decoder=decoder,
        pad_token_index=data.tgt_vocab['<pad>']
    )
    
    logger = TrainingLogger(
        log_path='logs/seq2seq_mt_experiment_gereng.json',
        hparams=hparams
    )
    
    optim = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    
    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        grad_clip=hparams['grad_clip'],
        optimizer=optim,
        save_path='./models/seq2seq_mt_gereng',
        logger=logger,
        device=torch.device('mps'),
    )
    
    def init_seq2seq(m):
        """Initialize weights for sequence-to-sequence learning."""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # model.apply(init_seq2seq)
    # trainer = Seq2SeqTrainer(model, train_loader, val_loader, config)
    # trainer.train()
    # logger.summary()
    
    model: Seq2Seq = load_model('./models/seq2seq_mt_gereng',
                                model, device=torch.device('cpu'))
    engs, des = data.test_sentences
    preds, _ = model.generate(
        data.build(engs, des),
        torch.device('cpu'),
        max_len=50)
    preds = preds.cpu().tolist()
    eval_translations(srcs=engs, dsts=des, preds=preds, data=data)