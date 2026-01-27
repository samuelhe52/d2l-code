import torch
from torch import nn, Tensor

def layer_norm(X: Tensor, gamma: Tensor, beta: Tensor,
               eps: float) -> Tensor:
    mean = X.mean(dim=-1, keepdim=True)
    var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)
    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = gamma * X_hat + beta
    return Y

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        shape = (1, num_features)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = 1e-5
        
    def forward(self, X: Tensor) -> Tensor:
        Y = layer_norm(X, self.gamma, self.beta, self.eps)
        return Y