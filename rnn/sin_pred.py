import torch
from torch import nn, Tensor
from utils.regression import train, validate
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class SinData(Dataset):
    """
    Synthetic dataset for sine wave prediction with noise.

    Args:
        T (int): Total number of time steps to generate.
        noise_std (float): Standard deviation of Gaussian noise added to the sine wave.
        seq_len (int): Length of input sequences.
        num_train (int): Number of training samples; the rest are for testing.
    """
    def __init__(self, T: int = 1000, noise_std: float = 0.1,
                 seq_len: int = 4, num_train: int = 600):
        self.seq_len = seq_len
        self.num_train = num_train
        time = torch.arange(0, T, dtype=torch.float32)
        self.x = torch.sin(0.01 * time) + torch.randn(T) * noise_std 

    def __len__(self):
        return len(self.x) - self.seq_len
    
    def __getitem__(self, idx):
        return (self.x[idx:idx + self.seq_len], # input sequence
                self.x[idx + self.seq_len]) # label (next value)

def get_dataloader(batch_size: int = 16, train: bool = True):
    """
    Build a DataLoader for sine wave prediction.

    Args:
        batch_size (int): Batch size for the DataLoader.
        train (bool): When True, return the training loader; otherwise return test loader.
    Returns:
        DataLoader: Torch DataLoader yielding (input_sequence, label) pairs.
    """
    dataset = SinData()
    if train:
        subset = torch.utils.data.Subset(dataset, range(dataset.num_train))
    else:
        subset = torch.utils.data.Subset(dataset, range(dataset.num_train, len(dataset)))
    return DataLoader(dataset=subset, batch_size=batch_size, shuffle=train)

class LinearRegression(nn.Module):
    """
    A simple linear regression model for sequence prediction.
    """
    def __init__(self, seq_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, X: Tensor) -> Tensor:
        return self.linear(X)
    
def k_step_ahead_predict(model: nn.Module, X: Tensor, k: int) -> Tensor:
    """
    Perform k-step ahead prediction using the model.

    Args:
        model (nn.Module): The trained model.
        X (Tensor): Input tensor of shape (batch_size, seq_len).
        k (int): Number of steps to predict ahead.
    Returns:
        Tensor: Predictions after k steps, shape (batch_size,).
    """
    model.eval()
    preds = []
    input_seq = X.clone()
    
    for _ in range(k):
        with torch.no_grad():
            pred = model(input_seq).squeeze()
        preds.append(pred)
        input_seq = torch.cat((input_seq[:, 1:], pred.unsqueeze(1)), dim=1)
    
    return torch.stack(preds, dim=1)

if __name__ == "__main__":
    batch_size = 16
    num_epochs = 10
    lr = 0.01

    train_loader = get_dataloader(batch_size=batch_size, train=True)
    test_loader = get_dataloader(batch_size=batch_size, train=False)

    model = LinearRegression(seq_len=4)
    train(model, train_loader, num_epochs, lr, device=torch.device('cpu'))
    
    import matplotlib.pyplot as plt
    model.eval()
    preds = []
    labels = []
    
    for X, y in test_loader:
        with torch.no_grad():
            pred = model(X).squeeze().numpy()
        preds.extend(pred)
        labels.extend(y.numpy())
        
    # Perform k-step ahead prediction
    k = 200
    preds_k_step = []
    labels_k_step = []
    
    for X, y in test_loader:
        with torch.no_grad():
            pred_k = k_step_ahead_predict(model, X, k)
        preds_k_step.extend(pred_k[:, -1].numpy())  # Take the last prediction
        labels_k_step.extend(y.numpy())
        
    # Visualize in two subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(labels, label='True Values')
    plt.plot(preds, label='1-step Predictions')
    plt.title('1-step Ahead Predictions')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(labels_k_step, label='True Values')
    plt.plot(preds_k_step, label=f'{k}-step Predictions')
    plt.title(f'{k}-step Ahead Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    # Evaluate on test data
    avg_loss = validate(model, loss_fn=nn.MSELoss(), dataloader=test_loader)
    print(f"Test MSE Loss: {avg_loss:.4f}")