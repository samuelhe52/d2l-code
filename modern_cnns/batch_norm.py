import torch
from torch import nn, Tensor
from typing import Tuple

def batch_norm(X: Tensor, gamma: Tensor, beta: Tensor,
               moving_mean: Tensor, moving_var: Tensor,
               eps: float, momentum: float, is_training: bool) -> Tuple[Tensor, Tensor, Tensor]:
    # Detect in training or inference
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4) # 2 for FC, 4 for convolution
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update moving mean and var
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    """
    Batch Normalization Layer for both Fully Connected and Convolutional layers.
    
    Args:
        num_features (int): Number of features or channels.
        num_dims (int): Dimensionality of the input (2 for FC, 4 for Conv).
    """
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2: # FC
            shape = (1, num_features)
        if num_dims == 4: # Convolution
            shape = (1, num_features, 1, 1) # num_features is C
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.register_buffer('moving_mean', self.moving_mean)
        self.register_buffer('moving_var', self.moving_var)
        self.eps = 1e-5
        self.momentum = 0.1
        
    def forward(self, X: Tensor) -> Tensor:
        if self.moving_mean.device != X.device: # Sanity check
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, moving_mean, moving_var = batch_norm(
            X, self.gamma, self.beta,
            self.moving_mean, self.moving_var,
            self.eps, self.momentum, self.training
        )
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        return Y
