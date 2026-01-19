import torch
from torch import nn
from utils.regression import train
from utils import TrainingLogger
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
    
class MLP(nn.Module):
    def __init__(self, hidden_units=[64, 32], dropout = 0.0):
        super().__init__()
        layers = []
        for hidden_unit in hidden_units:
            layers.append(nn.LazyLinear(hidden_unit))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.LazyLinear(1))  # Output layer
        self.net = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.net(X)
    
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
    preds = []
    for X in test_loader:
        pred = predict_price(models, X)
        preds.append(pred)
    tensor_preds = torch.cat(preds)
    return tensor_preds

def generate_submission(models, test_loader, path):
    """Generate submission file for Kaggle competition.
    
    Args:
        models: List of trained LinearRegression models
        test_loader: DataLoader for test dataset
        path: Path to the output CSV file
    """
    import pandas as pd

    predictions = test(models, test_loader)
    submission_df = pd.DataFrame({
        "Id": range(1461, 1461 + len(predictions)),  # test IDs start from 1461
        "SalePrice": predictions.numpy()
    })
    submission_df.to_csv(path, index=False)
    print(f"Submission file '{path}' generated.")


if __name__ == "__main__":
    hparams = {
        'model': 'MLP',
        'hidden_units': [32, 24, 16],
        'dropout': 0.00,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'num_epochs': 100,
        'lr': 0.01,
    }
    
    loaders = k_fold_loaders(k=6, batch_size=hparams['batch_size'])
    models = []
    
    logger = TrainingLogger(
        log_path='logs/house_price_mlp_kfold.json',
        hparams=hparams
    )

    tqdm.write("Starting k-fold cross-validation training...")
    bar = tqdm(range(len(loaders)), desc="K-Folds")
    
    for i, (train_loader, val_loader) in zip(bar, loaders, strict=True):
        tqdm.write(f"Training fold {i + 1}/{len(loaders)}")

        model = MLP(hidden_units=hparams['hidden_units'],
                    dropout=hparams['dropout'])
            
        optim = torch.optim.SGD(
            model.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay']
        )
        _, val_loss = train(
            model,
            train_loader,
            num_epochs=hparams['num_epochs'],
            lr=hparams['lr'],
            optimizer=optim,
            verbose=False,
            logger=logger if i == len(loaders) - 1 else None, # Log only last fold
            val_dataloader=val_loader,
            device=torch.device('cpu')
        )

        models.append((model, val_loss))
    
    models.sort(key=lambda x: x[1])  # Sort by validation loss
    models = [m[0] for m in models[:3]]  # Keep top 3 models
    logger.summary()
    logger.save()
    
    test_loader = get_dataloader(batch_size=64, train=False)
    # predictions = test(models, test_loader)
    # print("Predictions on test set:", predictions[:10])  # Print first 10 predictions
    
    generate_submission(models, test_loader, path="data/HousePrices/submission.csv")
        
    
        