"""Data loading utilities."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(batch_size, train=True, resize=(28, 28), data_root='./data'):
    """Download the Fashion-MNIST dataset and load it into memory.
    
    Args:
        batch_size: Batch size for the DataLoader
        train: If True, load training set; otherwise load test set
        resize: Tuple of (height, width) to resize images to
        data_root: Root directory for storing/loading the dataset
        
    Returns:
        DataLoader for Fashion-MNIST dataset
    """
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    dataset = datasets.FashionMNIST(
        root=data_root,
        train=train,
        transform=transform,
        download=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
