import torch
from d2l import torch as d2l

class LinearRegressionScratch(d2l.Module):
    """
    A linear regression model implemented from scratch.
    
    Parameters
    ----------
    num_inputs: int
        Number of input features.
    lr: float
        Learning rate for optimization.
    sigma: float, optional
        Standard deviation for initializing weights. Default is 0.01.
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    
    def forward(self, X):
        """
        Forward pass to compute predictions.
        
        Parameters
        ----------
        X: torch.Tensor
            Input features of shape (batch_size, num_inputs).
        """
        return torch.matmul(X, self.w) + self.b
    
    def loss(self, y_hat, y):
        """
        Compute mean squared error loss.
        
        Parameters
        ----------
        y_hat: torch.Tensor
            Predicted targets of shape (batch_size, 1).
        y: torch.Tensor
            True targets of shape (batch_size, 1).
        """
        # / 2 for convenient derivative
        loss_tensor = (y_hat - y) ** 2 / 2
        return loss_tensor.mean()
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], lr=self.lr)

class SGD(d2l.HyperParameters):
    """
    Stochastic Gradient Descent optimizer.
    
    Parameters
    ----------
    params: list[torch.Tensor]
        List of model parameters to optimize.
    lr: float
        Learning rate.
    """
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        """
        Perform a single optimization step.
        """
        for param in self.params:
            param.data -= self.lr * param.grad
            
    def zero_grad(self):
        """
        Zero the gradients of all optimized parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
