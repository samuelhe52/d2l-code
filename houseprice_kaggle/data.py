import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KaggleHouse(Dataset):
    """
    Dataset that yields `(features, labels)` pairs when labels are provided, otherwise features only.

    Args:
        features (pd.DataFrame): Preprocessed feature columns ready for tensor conversion.
        labels (Optional[pd.Series]): Matching target values; when None the dataset yields features only.
    Returns:
        KaggleHouse: Iterable dataset compatible with torch DataLoader.
    """

    def __init__(self, features, labels=None):
        self.features = torch.tensor(features.values.astype(float), dtype=torch.float32)
        self.labels = None
        if labels is not None:
            labels_tensor = torch.tensor(labels.values.astype(float), dtype=torch.float32).reshape(-1, 1)
            self.labels = torch.log(labels_tensor)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]


def preprocess(root='data/HousePrices/'):
    """
    Load raw CSVs, normalize numeric columns, and one-hot encode categoricals.

    Args:
        root (str): Directory containing `train.csv` and `test.csv`.
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: (train_features, labels, test_features) aligned by row order.
    """
    label = 'SalePrice'
    raw_train = pd.read_csv(root + 'train.csv')
    raw_test = pd.read_csv(root + 'test.csv')

    features = pd.concat(
        [raw_train.drop(columns=[label, 'Id']), raw_test.drop(columns=['Id'])],
        axis=0,
        ignore_index=True,
    )

    # Get numeric features
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    # Standardize numeric features
    features[numeric_features] = features[numeric_features].apply(
        lambda col: (col - col.mean()) / col.std()
    )
    # Fill missing numeric values with 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # One-hot encode categorical features
    features = pd.get_dummies(features, dummy_na=True)

    n_train = raw_train.shape[0]
    train_features = features.iloc[:n_train].reset_index(drop=True)
    test_features = features.iloc[n_train:].reset_index(drop=True)
    labels = raw_train[label].reset_index(drop=True)

    return train_features, labels, test_features


def _tensor_dataset(df, label_col='SalePrice'):
    """
    Convert a DataFrame with a label column into feature and label tensors.

    Args:
        df (pd.DataFrame): Data containing both features and the label column.
        label_col (str): Name of the label column to extract.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (features, log-transformed labels).
    """
    features = torch.tensor(df.drop(columns=[label_col]).values.astype(float), dtype=torch.float32)
    labels = torch.tensor(df[label_col].values.astype(float), dtype=torch.float32).reshape(-1, 1)
    labels = torch.log(labels)
    return features, labels


def get_dataloader(batch_size, train=True, root='data/HousePrices/', train_df=None, val_df=None):
    """
    Build a DataLoader for the train split, validation split, or Kaggle test set.

    Args:
        batch_size (int): Batch size for the DataLoader.
        train (bool): When True, return the training loader; otherwise return validation/test loader.
        root (str): Directory containing raw CSVs (used when custom folds are not supplied).
        train_df (Optional[pd.DataFrame]): Pre-split training rows with SalePrice column for custom folds.
        val_df (Optional[pd.DataFrame]): Pre-split validation rows with SalePrice column for custom folds.
    Returns:
        DataLoader: Torch DataLoader yielding (features, labels) for train/val or features for test.
    """

    train_features, labels, test_features = preprocess(root)

    if train_df is not None and val_df is not None:
        data = train_df if train else val_df
        features, lbls = _tensor_dataset(data)
        dataset = torch.utils.data.TensorDataset(features, lbls)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)

    if train:
        dataset = KaggleHouse(train_features, labels)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    dataset = KaggleHouse(test_features, labels=None)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


def k_fold_loaders(k, batch_size, root='data/HousePrices/', seed=42, shuffle=True):
    """
    Generate k folds of DataLoaders for cross-validation using row-wise splits.

    Args:
        k (int): Number of folds to create.
        batch_size (int): Batch size for each loader.
        root (str): Directory containing raw CSVs.
        seed (int): RNG seed for reproducible shuffling.
        shuffle (bool): Shuffle indices before splitting when True.
    Returns:
        List[Tuple[DataLoader, DataLoader]]: List of `(train_loader, val_loader)` pairs.
    """

    train_features, labels, _ = preprocess(root)
    full = train_features.copy()
    full['SalePrice'] = labels

    n = len(full)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    folds = np.array_split(idx, k)
    loaders = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i + 1:]) if k > 1 else val_idx
        train_df = full.iloc[train_idx].reset_index(drop=True)
        val_df = full.iloc[val_idx].reset_index(drop=True)
        train_loader = get_dataloader(batch_size, train=True, train_df=train_df, val_df=val_df)
        val_loader = get_dataloader(batch_size, train=False, train_df=train_df, val_df=val_df)
        loaders.append((train_loader, val_loader))

    return loaders
