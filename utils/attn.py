"""Toolset for attention-related operations."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def show_heatmap(attn_weights: Tensor, xlabel: str = 'Keys',
            ylabel: str = 'Queries', title: str = 'Attention Weights',
            cmap: str = 'Reds') -> None:
    """
    Plot a heatmap of attention weights.
    
    Args:
        attn_weights: Attention weights 4D tensor of shape
            (row, col, query_len, key_len). Subplots will be created for each
            (row, col) pair.
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        title: Title of the heatmap
        cmap: Colormap for the heatmap
    """
    row, col, _, _ = attn_weights.shape
    fig, axes = plt.subplots(row, col, figsize=(col * 3, row * 3),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax_row, weights_row) in enumerate(zip(axes, attn_weights)):
        for j, (ax, weights) in enumerate(zip(ax_row, weights_row)):
            im = ax.imshow(weights.detach().cpu(), cmap=cmap)
            ax.set_title(f'{title} ({i}, {j})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

class DotProductAttention(nn.Module):
    """
    Single-Head Scaled Dot Product Attention mechanism.

    Args:
        dropout: Dropout rate to apply on attention weights.
        optimized: Whether to use the optimized F.scaled_dot_product_attention.
            When True, leverages PyTorch's built-in function for efficiency,
            but attn_weights attribute will not be populated.
    """
    def __init__(self, dropout: float = 0.0, optimized: bool = True):
        super().__init__()
        self.optimized = optimized
        self.dropout = dropout

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Tensor = None) -> Tensor:
        """
        Compute the attention output.
        
        Args:
            queries: Query tensor of shape (batch_size, num_queries, d)
            keys: Key tensor of shape (batch_size, num_kv_pairs, d)
            values: Value tensor of shape (batch_size, num_kv_pairs, value_dim)
            valid_lens: Optional tensor indicating valid lengths for masking.
                Shape (batch_size,) or (batch_size, num_queries)
        Returns:
            Attention output tensor of shape (batch_size, num_queries, value_dim)
        """
        if self.optimized:
            # scaled_dot_product_attention performs multi-head attention,
            # we need to adapt our inputs to fit its expected shape.
            # attn_mask shape should be broadcastable to 
            # (batch_size, num_heads, num_queries, num_kv_pairs)
            queries = queries.unsqueeze(1)  # (batch_size, 1, num_queries, d)
            keys = keys.unsqueeze(1)        # (batch_size, 1, num_kv_pairs, d)
            values = values.unsqueeze(1)    # (batch_size, 1, num_kv_pairs, value_dim)
            mask = None
            if valid_lens is not None:
                if valid_lens.dim() == 1: # (batch_size,)
                    # (batch_size, num_queries)
                    valid_lens = valid_lens \
                        .unsqueeze(1) \
                        .expand(-1, queries.shape[2])
                mask = torch.arange(keys.shape[2], device=keys.device)[None, None, :] < valid_lens[:, :, None]
                mask = mask.unsqueeze(1)  # (batch_size, 1, num_queries, num_kv_pairs)
            return F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0
            ).squeeze(1)  # Remove the num_heads dimension
        else:
            d = queries.shape[-1]
            scores: Tensor = torch.bmm(queries, keys.transpose(1, 2)) / (d ** 0.5)
            if valid_lens is not None:
                if valid_lens.dim() == 1:
                    valid_lens = valid_lens[:, None].expand(-1, queries.shape[1])
                # (batch_size, num_queries, num_kv_pairs)
                mask = torch.arange(keys.shape[1], device=keys.device)[None, None, :] >= valid_lens[:, :, None]
                scores.masked_fill_(mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            return torch.bmm(attn_weights, values)
    
    


class AdditiveAttention(nn.Module):
    """
    Single-Head Additive Attention mechanism.

    Args:
        query_size: Feature size of each query vector.
        key_size: Feature size of each key vector.
        num_hiddens: Number of hidden units in the attention mechanism.
        dropout: Dropout rate to apply on attention weights.
    """
    def __init__(
        self,
        query_size: int,
        key_size: int,
        num_hiddens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Tensor = None) -> Tensor:
        """
        Compute the attention output.
        
        Args:
            queries: Query tensor of shape (batch_size, num_queries, query_size)
            keys: Key tensor of shape (batch_size, num_kv_pairs, key_size)
            values: Value tensor of shape (batch_size, num_kv_pairs, value_dim)
            valid_lens: Optional tensor indicating valid lengths for masking.
                Shape (batch_size,) or (batch_size, num_queries)
        Returns:
            Attention output tensor of shape (batch_size, num_queries, value_dim)
        """
        # Project queries and keys
        # queries: (batch_size, num_queries, num_hiddens)
        # keys: (batch_size, num_kv_pairs, num_hiddens)
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        
        # Expand dims for broadcasting: 
        # queries: (batch_size, num_queries, 1, num_hiddens)
        # keys: (batch_size, 1, num_kv_pairs, num_hiddens)
        # features: (batch_size, num_queries, num_kv_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # scores: (batch_size, num_queries, num_kv_pairs)
        scores = self.w_v(features).squeeze(-1)
        
        # Apply masking based on valid_lens
        if valid_lens is not None:
            if valid_lens.dim() == 1:  # (batch_size,)
                valid_lens = valid_lens[:, None].expand(-1, queries.shape[1])
            # mask: (batch_size, num_queries, num_kv_pairs)
            mask = torch.arange(keys.shape[1], device=keys.device)[None, None, :] >= valid_lens[:, :, None]
            scores.masked_fill_(mask, float('-inf'))
        
        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.bmm(attn_weights, values)


def valid_lens_to_key_padding_mask(valid_lens: Tensor, key_len: int) -> Tensor:
    """
    Convert valid lengths to `nn.MultiheadAttention` key padding mask.

    Args:
        valid_lens: Tensor of shape (batch_size,) representing valid key lengths
            per sample.
        key_len: Number of key/value positions.

    Returns:
        Boolean mask of shape (batch_size, key_len), where True indicates
        positions to be masked (ignored).
    """
    if valid_lens.dim() != 1:
        raise ValueError(
            f"valid_lens for key_padding_mask must be 1D, got shape {tuple(valid_lens.shape)}"
        )
    if key_len <= 0:
        raise ValueError(f"key_len must be positive, got {key_len}")
    if valid_lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"valid_lens must be integer dtype, got {valid_lens.dtype}")

    return torch.arange(key_len, device=valid_lens.device)[None, :] >= valid_lens[:, None]


def valid_lens_to_attn_mask(valid_lens: Tensor, query_len: int, key_len: int,
                            num_heads: int = 1) -> Tensor:
    """
    Convert valid lengths to `nn.MultiheadAttention` 3D attention mask.

    Args:
        valid_lens: Tensor of shape (batch_size, query_len), where each element
            is the valid key length for a given (sample, query_position).
        query_len: Number of query positions.
        key_len: Number of key/value positions.
        num_heads: Number of attention heads in MHA.

    Returns:
        Boolean attention mask of shape
        (batch_size * num_heads, query_len, key_len), where True indicates
        positions to be masked (ignored).
    """
    if valid_lens.dim() != 2:
        raise ValueError(
            f"valid_lens for attn_mask must be 2D, got shape {tuple(valid_lens.shape)}"
        )
    if valid_lens.shape[1] != query_len:
        raise ValueError(
            f"query_len mismatch: valid_lens has {valid_lens.shape[1]}, got {query_len}"
        )
    if key_len <= 0:
        raise ValueError(f"key_len must be positive, got {key_len}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if valid_lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"valid_lens must be integer dtype, got {valid_lens.dtype}")

    mask = torch.arange(key_len, device=valid_lens.device)[None, None, :] >= valid_lens[:, :, None]
    mask = mask[:, None, :, :].expand(-1, num_heads, -1, -1)
    return mask.reshape(-1, query_len, key_len)


class MultiheadAttentionWithValidLens(nn.Module):
    """
    Thin wrapper around `nn.MultiheadAttention` with `valid_lens` support.

    Accepts D2L-style `valid_lens` and converts it to the masks expected by
    `nn.MultiheadAttention`.

    - 1D `valid_lens` of shape (batch_size,) -> `key_padding_mask`
    - 2D `valid_lens` of shape (batch_size, query_len) -> 3D `attn_mask`
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                valid_lens: Optional[Tensor] = None,
                need_weights: bool = False,
                average_attn_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            Tuple[Tensor, Optional[Tensor]]: Output tensor of shape (batch_size, query_len, embed_dim) and
            attention weights of shape (batch_size, num_heads, query_len, key_len) if `need_weights` is True, otherwise None.
        """
        if self.batch_first:
            batch_size, query_len = query.shape[0], query.shape[1]
            key_len = key.shape[1]
        else:
            query_len, batch_size = query.shape[0], query.shape[1]
            key_len = key.shape[0]

        key_padding_mask = None
        derived_attn_mask = None

        if valid_lens is not None:
            if valid_lens.shape[0] != batch_size:
                raise ValueError(
                    f"batch mismatch: query batch={batch_size}, valid_lens batch={valid_lens.shape[0]}"
                )
            if valid_lens.dim() == 1:
                key_padding_mask = valid_lens_to_key_padding_mask(valid_lens, key_len)
            elif valid_lens.dim() == 2:
                derived_attn_mask = valid_lens_to_attn_mask(
                    valid_lens, query_len=query_len, key_len=key_len,
                    num_heads=self.num_heads
                )
            else:
                raise ValueError(
                    f"valid_lens must be 1D or 2D, got shape {tuple(valid_lens.shape)}"
                )

        if attn_mask is not None and derived_attn_mask is not None:
            if attn_mask.dtype != torch.bool or derived_attn_mask.dtype != torch.bool:
                raise TypeError(
                    "When combining attn_mask with valid_lens-derived mask, both masks must be bool"
                )
            attn_mask = attn_mask | derived_attn_mask
        elif derived_attn_mask is not None:
            attn_mask = derived_attn_mask

        return self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )


class SinusoidalPositionalDecoding(nn.Module):
    """
    Sinusoidal positional decoding for token embeddings.

    This module precomputes fixed sinusoidal positional vectors and adds them
    to the input embedding tensor, then applies dropout.

    Args:
        num_hiddens: Embedding/hidden dimension.
        dropout: Dropout probability applied after adding positional vectors.
        max_len: Maximum sequence length supported by the precomputed table.
    """
    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Positional table with shape (1, max_len, num_hiddens).
        P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens,
        )
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.register_buffer('P', P)

    def forward(self, X: Tensor, offset: int = 0) -> Tensor:
        """
        Add fixed sinusoidal positional encodings to input embeddings.

        Args:
            X: Input embeddings of shape (batch_size, seq_len, num_hiddens).
            offset: Starting position index. Use this during incremental
                decoding so that step *t* receives position *t*'s encoding
                rather than always position 0.

        Returns:
            Tensor of shape (batch_size, seq_len, num_hiddens).
        """
        X = X + self.P[:, offset:offset + X.shape[1], :]
        return self.dropout(X)
    
