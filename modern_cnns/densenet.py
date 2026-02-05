import torch
from torch import nn, Tensor
from torch.nn import functional as F
from utils.data import fashion_mnist
from utils.training import ClassificationTrainer, TrainingLogger, TrainingConfig
from typing import Dict, Any, Type, List, Tuple

class DenseBlock(nn.Module):
    """
    A dense block consisting of multiple convolutional blocks.
    
    Args:
        num_convs (int): Number of convolutional blocks in this dense block.
        growth_rate (int): Growth rate (number of output channels) for each conv block.
    """
    def __init__(self, num_convs: int, growth_rate: int):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(self.conv_block(growth_rate))
        self.net = nn.ModuleList(layers)
        
    def forward(self, X: Tensor) -> Tensor:
        for blk in self.net:
            out = blk(X)
            X = torch.cat((X, out), dim=1)
        return X

    def conv_block(self, num_channels: int) -> nn.Sequential:
        """
        A convolutional block consisting of BatchNorm, ReLU, and Conv2d.

        Args:
            num_channels (int): Number of output channels for the convolution.
        """
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(num_channels, kernel_size=3, padding=1, bias=False),
        )


class DenseNet(nn.Module):
    """
    A DenseNet model.
    
    Args:
        stem_channels (int): Number of output channels for the initial convolutional layer.
        arch (List[int]): List specifying the number of conv blocks in each dense block.
        growth_rate (int): Growth rate (number of output channels) for each conv block.
        num_classes (int): Number of output classes for classification.
    """
    def __init__(self, stem_channels: int, arch: List[int],
                 growth_rate = 32, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            self.stem(stem_channels),
        )
        num_channels = stem_channels

        for i, num_convs in enumerate(arch):
            self.net.add_module(
                f'denseblock_{i+1}', DenseBlock(num_convs, growth_rate)
            )
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1: # If not at the last dense block
                num_channels //= 2  # Due to transition layer
                self.net.add_module(
                    f'transition_{i+1}',
                    self.transition_block(num_channels)
                )
                
        self.net.add_module('head', self.head(num_classes))
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
    
    def stem(self, num_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    
    def head(self, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(), # Account for the last block's pre-activation
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes),
        )
        
    def transition_block(self, num_channels: int) -> nn.Sequential:
        """
        A transition block that reduces the number of channels and halves the spatial dimensions.
        
        Args:
            num_channels (int): Number of output channels for the convolution.
        """
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(num_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2), # Halve the spatial dimensions
        )

class DenseNet18(DenseNet):
    def __init__(self, num_classes: int = 10):
        arch = [2, 2, 2, 2]  # Number of conv blocks in each dense block
        super().__init__(stem_channels=64,
                         arch=arch,
                         growth_rate=32,
                         num_classes=num_classes)
    
if __name__ == "__main__":
    batch_size = 128
    num_epochs = 20 # 10 is mostly enough, 20 used for experiments
    lr = 0.05
    
    model = DenseNet18(num_classes=10)
    dataloader = fashion_mnist(batch_size, resize=96, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, resize=96, data_root='data/')
    init_fn = torch.nn.init.kaiming_uniform_
    
    hparams: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'init_fn': init_fn.__name__,
    }
    
    logger = TrainingLogger(
        log_path='logs/densenet18_experiment.json',
        hparams=hparams
    )
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init_fn(m.weight)
    
    model.forward(torch.randn(1, 1, 96, 96))  # Initialize lazy layers
    model.apply(init_weights)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        save_path='models/densenet18.pt',
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
    