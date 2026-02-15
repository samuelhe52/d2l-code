import torch
from torch import nn
import torch.nn.functional as F
from utils.attn import (
    MultiheadAttentionWithValidLens,
    SinusoidalPositionalDecoding
)
from utils.enc_dec import Encoder, Decoder, EncoderDecoder
from utils.io import load_model
from utils.training import (
    TrainingConfig, TrainingLogger,
    Seq2SeqTrainer
)
from utils.data.mt_data import (
    GerEngDataset, FraEngDataset,
    mt_dataloader, eval_translations
)

from typing import Optional, Tuple

class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Two linear transformations with a ReLU in between:
    FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2

    Args:
        num_hiddens: Model dimension (d_model); also the output dimension.
        ffn_num_hiddens: Inner / intermediate dimension (d_ff).
    """
    def __init__(self, num_hiddens: int, ffn_num_hiddens: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens, num_hiddens)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.ffn(X)
    
    
class AddNorm(nn.Module):
    """
    Add & Norm layer.

    Args:
        normalized_shape: The shape of the input to be normalized.
        dropout: Dropout rate.
    """
    def __init__(self, normalized_shape: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.norm(X + self.dropout(Y))
    

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer. Consists of two sub-layers:
    1. Multi-head self-attention mechanism.
    2. Position-wise feed-forward network.

    Args:
        num_hiddens: Number of hidden units in the encoder layer.
        ffn_num_hiddens: Number of hidden units in the feed-forward network.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        bias: Whether to use bias in the linear layers.
    """
    def __init__(self, num_hiddens: int, ffn_num_hiddens: int,
                 num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        self.attention = MultiheadAttentionWithValidLens(
            num_hiddens, num_heads, dropout, bias
        )
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionwiseFFN(num_hiddens, ffn_num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        
    
    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor) \
        -> torch.Tensor:
        # nn.MultiheadAttention returns (attn_output, attn_weights), so we
        # explicitly take the first element for residual connection.
        self_attn_out = self.attention(X, X, X, valid_lens)[0]
        Y = self.addnorm1(X, self_attn_out)
        return self.addnorm2(Y, self.ffn(Y))
    
    
class TransformerEncoder(Encoder):
    """
    Transformer Encoder.

    Args:
        vocab_size: Size of the vocabulary.
        num_hiddens: Number of hidden units in the encoder.
        ffn_num_hiddens: Number of hidden units in the feed-forward network.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout rate.
        bias: Whether to use bias in the linear layers.
    """
    def __init__(self, vocab_size: int, num_hiddens: int,
                 ffn_num_hiddens: int, num_heads: int,
                 num_layers: int, dropout: float, bias: bool = False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = SinusoidalPositionalDecoding(num_hiddens, dropout)
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerEncoderLayer(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, bias
            ))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor) \
        -> torch.Tensor:
        # Scale token embeddings by sqrt(d_model), as in the original Transformer.
        X = self.pos_encoding(self.embedding(X) * (self.num_hiddens ** 0.5))
        for layer in self.layers:
            X = layer(X, valid_lens)
        return X
    

class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer. Consists of three sub-layers:
    1. Masked multi-head self-attention mechanism.
    2. Multi-head attention mechanism over the encoder's output.
    3. Position-wise feed-forward network.
    
    During inference this layer caches **pre-projection hidden states**
    (not the post-projection K/V tensors) to avoid recomputing earlier
    layers.  ``nn.MultiheadAttention`` still re-projects the full cached
    sequence through W_K and W_V at every step.

    Args:
        num_hiddens: Number of hidden units in the decoder layer.
        ffn_num_hiddens: Number of hidden units in the feed-forward network.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        bias: Whether to use bias in the linear layers.
    """
    def __init__(self, num_hiddens: int, ffn_num_hiddens: int,
                 num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        self.attention1 = MultiheadAttentionWithValidLens(
            num_hiddens, num_heads, dropout, bias
        )
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiheadAttentionWithValidLens(
            num_hiddens, num_heads, dropout, bias
        )
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionwiseFFN(num_hiddens, ffn_num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)
        
    def forward(
        self,
        X: torch.Tensor,
        enc_outputs: torch.Tensor,
        enc_valid_lens: Optional[torch.Tensor],
        dec_valid_lens: Optional[torch.Tensor],
        hidden_cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: Input tensor of shape (batch_size, seq_len, num_hiddens).
            enc_outputs: Encoder outputs of shape
                (batch_size, src_seq_len, num_hiddens).
            enc_valid_lens: Valid lengths for encoder outputs.
            dec_valid_lens: Valid lengths for decoder self-attention keys.
                Used during training to mask decoder padding tokens.
            hidden_cache: Cached **pre-projection** decoder hidden states
                for this layer, shape (batch_size, cached_len, num_hiddens).
                None during training.  Note: these are fed as key/value
                inputs to ``nn.MultiheadAttention``, which re-applies W_K
                and W_V projections each step.

        Returns:
            Tuple of:
                - Decoder layer output of shape (batch_size, seq_len, num_hiddens)
                - Updated cache tensor of shape
                  (batch_size, cached_len + seq_len, num_hiddens)
        """
        # Self-attention branch:
        # - training: no cache, hidden states come from current sequence
        # - inference: append current step(s) to cached hidden states and
        #   attend over the full history. Note: nn.MultiheadAttention
        #   re-projects cached_hiddens through W_K / W_V every step.
        if hidden_cache is None:
            cached_hiddens = X
            query_len = X.shape[1]
            key_len = cached_hiddens.shape[1]
            causal_attn_mask = torch.triu(
                torch.ones(query_len, key_len, device=X.device, dtype=torch.bool),
                diagonal=1,
            )
            self_attn_valid_lens = dec_valid_lens
        else:
            cached_hiddens = torch.cat([hidden_cache, X], dim=1)
            # During incremental decoding, cached_hiddens only contains
            # past + current tokens, so no causal mask is needed.
            causal_attn_mask = None
            self_attn_valid_lens = None

        # 1) Masked decoder self-attention.
        self_attn_out = self.attention1(
            X,
            cached_hiddens,
            cached_hiddens,
            valid_lens=self_attn_valid_lens,
            attn_mask=causal_attn_mask,
            is_causal=False,
        )[0] # We only need the attention output
        Y = self.addnorm1(
            X,
            self_attn_out,
        )
        
        # 2) Encoder-decoder cross attention.
        # Y serves as the query, and enc_outputs serve as the key and value.
        # Attends to encoder outputs here.
        cross_attn_out = self.attention2(
            Y, enc_outputs, enc_outputs, enc_valid_lens
        )[0]
        Z = self.addnorm2(
            Y,
            cross_attn_out,
        )

        # 3) Position-wise feed-forward network.
        out = self.addnorm3(Z, self.ffn(Z))
        return out, cached_hiddens


class TransformerDecoder(Decoder):
    """Transformer decoder with per-layer hidden-state cache for incremental decoding."""

    def __init__(self, vocab_size: int, num_hiddens: int,
                 ffn_num_hiddens: int, num_heads: int,
                 num_layers: int, dropout: float, bias: bool = False):
        super().__init__()
        self._vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = SinusoidalPositionalDecoding(num_hiddens, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                num_hiddens=num_hiddens,
                ffn_num_hiddens=ffn_num_hiddens,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_layers)
        ])
        self.dense = nn.Linear(num_hiddens, vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def preprocess_state(
        self,
        enc_outputs: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list[Optional[torch.Tensor]]]:
        # State is a tuple so generic beam-search state selection utilities
        # can recursively gather/reorder tensor entries.
        enc_valid_lens = args[0] if len(args) > 0 else None
        hidden_cache = [None for _ in range(self.num_layers)]
        return enc_outputs, enc_valid_lens, hidden_cache

    def forward(
        self,
        X: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor], list[Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], list[Optional[torch.Tensor]]]]:
        enc_outputs, enc_valid_lens, hidden_cache = state

        token_ids = X.to(torch.long)
        dec_valid_lens: Optional[torch.Tensor] = None
        pad_token_index = getattr(self, 'pad_token_index', None)
        if pad_token_index is not None and hidden_cache[0] is None:
            dec_valid_lens = (token_ids != pad_token_index).sum(dim=1).to(torch.long)

        # Compute positional offset from hidden-state cache length so that
        # step t receives position t's encoding during incremental decoding.
        pos_offset = hidden_cache[0].shape[1] if hidden_cache[0] is not None else 0

        # Embed + positional encoding for decoder token ids.
        X = self.pos_encoding(
            self.embedding(token_ids) * (self.num_hiddens ** 0.5),
            offset=pos_offset,
        )

        new_hidden_cache: list[Optional[torch.Tensor]] = []
        for layer, layer_cache in zip(self.layers, hidden_cache):
            X, updated_cache = layer(
                X,
                enc_outputs=enc_outputs,
                enc_valid_lens=enc_valid_lens,
                dec_valid_lens=dec_valid_lens,
                hidden_cache=layer_cache,
            )
            new_hidden_cache.append(updated_cache)

        logits = self.dense(X).permute(0, 2, 1)
        return logits, (enc_outputs, enc_valid_lens, new_hidden_cache)


class TransformerSeq2Seq(EncoderDecoder):
    """Transformer-based encoder-decoder model."""

    def __init__(self,
                 encoder: TransformerEncoder,
                 decoder: TransformerDecoder,
                 pad_token_index: int,
                 eos_token_index: int | None = None):
        super().__init__(encoder, decoder)
        self.pad_token_index = pad_token_index
        self.decoder.pad_token_index = pad_token_index
        self.eos_token_index = eos_token_index


if __name__ == "__main__":
    hparams = {
        'seq_len': 25,
        'batch_size': 256,
        'num_epochs': 20,
        'lr': 1e-3,
        'grad_clip': 1.0,
        'num_hiddens': 256,
        'ffn_num_hiddens': 512,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.3,
        'bias': False,
    }

    data = GerEngDataset(
        seq_len=hparams['seq_len'],
        token_min_freq=3,
        # total_samples=20000,
    )
    train_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=False
    )

    shared = {
        'num_hiddens': hparams['num_hiddens'],
        'ffn_num_hiddens': hparams['ffn_num_hiddens'],
        'num_heads': hparams['num_heads'],
        'num_layers': hparams['num_layers'],
        'dropout': hparams['dropout'],
        'bias': hparams['bias'],
    }

    encoder = TransformerEncoder(vocab_size=len(data.src_vocab), **shared)
    decoder = TransformerDecoder(vocab_size=len(data.tgt_vocab), **shared)
    model = TransformerSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_token_index=data.tgt_vocab['<pad>'],
        eos_token_index=data.tgt_vocab['<eos>'],
    )

    logger = TrainingLogger(
        log_path='logs/transformer_mt_gereng.json',
        hparams=hparams,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        grad_clip=hparams['grad_clip'],
        optimizer=optimizer,
        save_path='./models/transformer_mt_gereng',
        logger=logger,
    )

    def init_transformer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    model.apply(init_transformer)

    trainer = Seq2SeqTrainer(model, train_loader, val_loader, config)
    trainer.train()
    logger.summary()

    model: TransformerSeq2Seq = load_model(
        './models/transformer_mt_gereng',
        model,
    )
    engs, des = data.test_sentences
    preds, _ = model.generate(
        data.build(engs, des),
        max_len=50,
        decode_strategy='beam',
        beam_size=4,
    )
    preds = preds.cpu().tolist()

    eval_translations(srcs=engs, dsts=des, preds=preds, data=data)
