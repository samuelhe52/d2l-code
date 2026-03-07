"""Classification data loading utilities."""

from typing import Sequence, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def fashion_mnist(
    batch_size: int,
    train: bool = True,
    resize: Sequence[int] | Tuple[int, int] = (28, 28),
    data_root: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Download the Fashion-MNIST dataset and load it into memory.

    Args:
        batch_size: Batch size for the DataLoader
        train: If True, load training set; otherwise load test set
        resize: Tuple of (height, width) to resize images to
        data_root: Root directory for storing/loading the dataset
        num_workers: Number of worker processes for the DataLoader
        pin_memory: Whether to pin host memory for faster CUDA transfers
        persistent_workers: Keep worker processes alive across epochs
        prefetch_factor: Number of batches prefetched per worker

    Returns:
        DataLoader for Fashion-MNIST dataset
    """
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    dataset = datasets.FashionMNIST(
        root=data_root, train=train, transform=transform, download=True
    )
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **dataloader_kwargs)
