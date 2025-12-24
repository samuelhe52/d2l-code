import torch
from torch import nn
from utils.regression import train
from tqdm import tqdm

try:
    # When run as a module
    from .data import k_fold_loaders, get_dataloader
except ImportError:  # pragma: no cover - fallback for direct script execution
    from data import k_fold_loaders, get_dataloader


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(1)

    def forward(self, x):
        return self.linear(x)
    
def predict_price(models, X):
    """Predict house prices using an ensemble of models.
    
    Args:
        models: List of trained LinearRegression models
        X: Input features tensor
    
    Returns:
        Tensor of predicted house prices
    """
    preds = [model(X).detach() for model in models]
    # Need to exponentiate since model predicts log-prices
    return torch.exp(torch.cat(preds, dim=1).mean(dim=1))

def test(models, test_loader):
    """Evaluate ensemble of models on test data.
    
    Args:
        models: List of trained LinearRegression models
        test_loader: DataLoader for test dataset
    
    Returns:
        Tensor of predicted house prices for the test set
    """
    all_preds = []
    for X in test_loader:
        preds = predict_price(models, X)
        all_preds.append(preds)
    tensor_preds = torch.cat(all_preds, dim=0)
    return tensor_preds

def generate_submission(models, test_loader, filename="submission.csv"):
    """Generate submission file for Kaggle competition.
    
    Args:
        models: List of trained LinearRegression models
        test_loader: DataLoader for test dataset
        filename: Name of the output CSV file
    """
    import pandas as pd

    predictions = test(models, test_loader)
    submission_df = pd.DataFrame({
        "Id": range(1461, 1461 + len(predictions)),  # Assuming test IDs start from 1461
        "SalePrice": predictions.numpy()
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission file '{filename}' generated.")


if __name__ == "__main__":
    loaders = k_fold_loaders(k=5, batch_size=64)
    models = []

    tqdm.write("Starting k-fold cross-validation training...")
    bar = tqdm(range(len(loaders)), desc="K-Folds")
    for i, (train_loader, val_loader) in zip(bar, loaders):
        tqdm.write(f"Training fold {i + 1}/{len(loaders)}")

        model = LinearRegression()
        train(
            model,
            train_loader,
            num_epochs=15,
            lr=0.01,
            print_epoch=False,
            val_dataloader=val_loader,
        )

        models.append(model)
        
    # test_loader = get_dataloader(batch_size=64, train=False)
    # predictions = test(models, test_loader)
    # print("Predictions on test set:", predictions[:10])  # Print first 10 predictions
    
    # generate_submission(models, test_loader, filename="submission.csv")
        
    
        