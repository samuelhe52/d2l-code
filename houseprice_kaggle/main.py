import torch
from torch import nn
import pandas as pd
from torch.utils.data import TensorDataset, Dataset, DataLoader

class KaggleHouse(Dataset):
    """
    Dataset class for Kaggle House Prices dataset.
    
    Args:
        root (str): Root directory of the dataset.
        train (pd.DataFrame, optional): Preloaded training data.
        val (pd.DataFrame, optional): Preloaded validation data.
    """
    def __init__(self, root='data/HousePrices/',
                 train=None, val=None):
        self.root = root
        self.train = train
        self.val = val
        
        if train is None:
            self.raw_train = pd.read_csv(root + 'train.csv')
            self.raw_val = pd.read_csv(root + 'test.csv')

    def __iter__(self):
        for idx in range(len(self.train)):
            yield self.train.iloc[idx]
    
    def preprocess(self):
        """
        Preprocess the dataset by handling missing values,
        encoding categorical variables, and normalizing numerical features.
        Sets self.train and self.val attributes.
        """
        # Remove the ID and SalePrice columns
        label = 'SalePrice'
        features = pd.concat(
            [self.raw_train.drop(columns=[label, 'Id']),
             self.raw_val.drop(columns=['Id'])])
        # Select all numeric features
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        # Normalize numeric features (zero mean, unit variance)
        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )
        # Replace NaN with 0
        features[numeric_features] = features[numeric_features].fillna(0)
        # One-hot encode categorical features
        features = pd.get_dummies(features, dummy_na=True)
        
        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        self.val = features[self.raw_train.shape[0]:].copy()
        
def get_dataloader(batch_size, train=True, root='data/HousePrices/'):
    """
    Create DataLoader for Kaggle House Prices dataset.
    
    Args:
        batch_size (int): Batch size for DataLoader.
        root (str): Root directory of the dataset.
    """
    dataset = KaggleHouse(root=root)
    dataset.preprocess()
    data = dataset.train if train else dataset.val
    get_tensor = lambda df: torch.tensor(df.values.astype(float),
                                         dtype=torch.float32)
    
    features = get_tensor(data.drop(columns=['SalePrice']))
    # log transform the labels
    labels = torch.log(get_tensor(data['SalePrice'])).reshape(-1, 1)
    return DataLoader(
        dataset=TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=train
    )
    
        
if __name__ == "__main__":
    data = KaggleHouse()
    data.preprocess()
    print(data.train.shape)