import torch
from torch import Tensor
from utils.attn import show_heatmap, DotProductAttention

def test_heatmap():
    weights = torch.eye(10).reshape(1, 1, 10, 10) + 0.05 * torch.randn(1, 1, 10, 10)
    show_heatmap(weights)

if __name__ == "__main__":
    test_heatmap()
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attention(queries, keys, values, valid_lens)
    