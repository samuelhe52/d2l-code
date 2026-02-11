"""Toolset for attention-related operations."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        query_size: Size of the query vectors.
        key_size: Size of the key vectors.
        num_hiddens: Number of hidden units in the attention mechanism.
        dropout: Dropout rate to apply on attention weights.
    """
    def __init__(self, query_size: int, key_size: int,
                 num_hiddens: int, dropout: float = 0.0):
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
    