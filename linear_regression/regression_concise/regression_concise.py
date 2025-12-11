import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression(nn.Module):
    """
    A concise linear regression model
    """
    def __init__(self, num_inputs):
        super().__init__()
        self.net = nn.Linear(num_inputs, 1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        
    def forward(self, X):
        return self.net(X)

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.randn(num_examples, len(w))
    y = X @ w + b
    y += torch.randn(y.shape) * 0.01  # Add noise
    return X, y.reshape(-1, 1)

def get_dataloader(X, y, batch_size, shuffle=True):
    """Create a DataLoader from features and labels."""
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, dataloader, num_epochs, lr):
    """Train the model."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
    
def get_w_and_b(model):
    w = model.net.weight.data
    b = model.net.bias.data
    return w, b

if __name__ == "__main__":
    train_nums = [5, 10, 100, 1000, 10000]
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    
    w_errors = []
    b_errors = []
    for n in train_nums:
        print(f"Training with {n} examples")
        # Create a fresh model for each training run
        model = LinearRegression(num_inputs=len(true_w))
        
        # Generate synthetic data
        X, y = synthetic_data(true_w, true_b, n)
        dataloader = get_dataloader(X, y, batch_size=min(n, 32))
        
        # Train the model
        train(model, dataloader, num_epochs=5, lr=0.03)
        
        w, b = get_w_and_b(model)
        
        error_w = torch.norm(w - true_w)
        error_b = torch.abs(b - true_b)
        w_errors.append(error_w.item())
        b_errors.append(error_b.item())
    
    plt.figure()
    plt.plot(train_nums, w_errors, marker='o', label='Weight Error')
    plt.plot(train_nums, b_errors, marker='s', label='Bias Error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.xscale('log')
    plt.title('Parameter Error vs Training Set Size')
    plt.show()
    plt.show()