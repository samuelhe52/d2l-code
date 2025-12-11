import random
import numpy as np
import torch
from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule):
    """
    Synthetic dataset for linear regression.

    Parameters
    ----------
    w: torch.Tensor
        Weights for the linear model.
    b: float
        Bias/intercept for the linear model.
    noise: float
        Standard deviation of Gaussian noise added to the targets.
    num_train: int
        Number of training samples.
    num_val: int
        Number of validation samples.
    batch_size: int
        Batch size for data loading.
    """
    def __init__(self, w, b,
                 noise=0.01,
                 num_train=1000,
                 num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w)) # n x num_features
        noise_tensor = torch.randn(n, 1) * noise # n x 1
        # Generate targets with noise. We store training and validation data together.
        self.y = torch.matmul(self.X, torch.tensor(w).reshape(-1, 1)) + b + noise_tensor

    def get_dataloader(self, train):
        """
        Get data loader for training or validation set.
        
        Parameters
        ----------
        train: bool
            If True, return training data loader; else return validation data loader.
            
        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for the specified dataset split.
        """
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        # Construct a data loader.
        # The data set is in the form of (features (X), targets (y)).
        return self.get_tensorloader((self.X, self.y), train, indices=i)
