import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Iterable
from utils import load_model
from utils.classfication import fashion_mnist
from pathlib import Path

# Add this file's folder to the path so we can import the model next to it.
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from lenet_modern import LeNetModern

def visualize_filters(model: nn.Module, layer_index: int = 0) -> None:
    """Visualize the filters of a specific convolutional layer."""
    conv_layer = None
    conv_count = 0
    for layer in model.net:
        if isinstance(layer, nn.Conv2d):
            if conv_count == layer_index:
                conv_layer = layer
                break
            conv_count += 1

    if conv_layer is None:
        raise ValueError(f"No convolutional layer found at index {layer_index}")

    filters = conv_layer.weight.data.cpu().numpy()
    num_filters = filters.shape[0]

    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters, 1))
    for i in range(num_filters):
        ax = axes[i]
        ax.imshow(filters[i, 0, :, :], cmap='gray')
        ax.axis('off')
    plt.show()

def visualize_activations(model: nn.Module, dataloader: Iterable, layer_index: int = 0) -> None:
    """Visualize activations of a convolutional layer for the entire batch."""
    conv_layer = None
    conv_count = 0
    for layer in model.net:
        if isinstance(layer, nn.Conv2d):
            if conv_count == layer_index:
                conv_layer = layer
                break
            conv_count += 1

    if conv_layer is None:
        raise ValueError(f"No convolutional layer found at index {layer_index}")

    # Get a single batch of data
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(next(model.parameters()).device)

    # Forward pass up to the specified layer
    x = inputs
    for layer in model.net:
        x = layer(x)
        if layer == conv_layer:
            break

    activations = x.data.cpu().numpy()
    originals = inputs.data.cpu().numpy()
    batch_size, num_activations, h, w = activations.shape

    # Add one column for the original image
    fig, axes = plt.subplots(batch_size, num_activations + 1,
                             figsize=(num_activations + 1, batch_size))

    # Ensure axes is 2D for consistent indexing
    if batch_size == 1:
        axes = axes[None, :]
    if num_activations + 1 == 1:
        axes = axes[:, None]

    for b in range(batch_size):
        axes[b, 0].imshow(originals[b, 0, :, :], cmap='gray')
        axes[b, 0].set_title('input')
        axes[b, 0].axis('off')

        for i in range(num_activations):
            ax = axes[b, i + 1]
            ax.imshow(activations[b, i, :, :], cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

model_path = 'models/lenet.pt'
model = LeNetModern()
load_model(model, model_path)
model.eval()
dataloader = fashion_mnist(batch_size=10, train=False, data_root='data/')

    
if __name__ == "__main__":
    # visualize_filters(model, layer_index=0)
    visualize_activations(model, dataloader, layer_index=1)