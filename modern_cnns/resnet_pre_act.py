# This implementation is an adaptation of ResNet with pre-activation residual blocks.

import torch
from torch import nn, Tensor
from utils.data import fashion_mnist
from utils.training import (
    ClassificationTrainer,
    TrainingLogger,
    TrainingConfig,
)
from typing import Dict, Any, Type, List, Tuple

class ResidualBasicBlock(nn.Module):
    """
    A basic residual block for ResNet.
    
    Args:
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolutional layer.
        use_proj (bool): Whether to use a projection shortcut.
    """
    def __init__(self, out_channels: int, stride: int = 1, use_proj: bool = True):
        super().__init__()
        self.norm_act = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        conv1 = nn.LazyConv2d(out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        conv2 = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.use_proj = use_proj
        self.conv_layers = nn.Sequential(conv1, conv2)
        self.proj = nn.LazyConv2d(out_channels, kernel_size=1,
                                  stride=stride, bias=False) if use_proj else None

    def forward(self, X: Tensor) -> Tensor:
        if self.use_proj:
            X = self.norm_act(X)
            out = self.conv_layers(X)
            shortcut = self.proj(X)
        else:
            out = self.conv_layers(self.norm_act(X))
            shortcut = X
        return out + shortcut

class ResidualBottleneckBlock(nn.Module):
    """
    A bottleneck residual block for ResNet.
    
    Args:
        mid_channels (int): Bottleneck width before expansion.
        stride (int): Stride for the second convolutional layer.
        expansion (int): Channel expansion factor applied after the third conv.
        use_proj (bool): Whether to use a projection shortcut.
    """
    def __init__(self, mid_channels: int, stride: int = 1, expansion: int = 4, use_proj: bool = True):
        super().__init__()
        out_channels = mid_channels * expansion
        self.norm_act = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        conv1 = nn.LazyConv2d(mid_channels, kernel_size=1, bias=False)
        conv2 = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        )
        conv3 = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
        )
        self.use_proj = use_proj
        self.proj = nn.LazyConv2d(out_channels, kernel_size=1,
                                  stride=stride, bias=False) if use_proj else None
        self.conv_layers = nn.Sequential(conv1, conv2, conv3)

    def forward(self, X: Tensor) -> Tensor:
        if self.use_proj:
            X = self.norm_act(X)
            out = self.conv_layers(X)
            shortcut = self.proj(X)
        else:
            out = self.conv_layers(self.norm_act(X))
            shortcut = X
        return out + shortcut

class ResNetStage(nn.Module):
    """
    A stage consisting of multiple residual blocks.
    
    Args:
        block_cls (nn.Module): The residual block class to use in this stage.
        mid_channels (int): Number of channels in the residual blocks.
            For bottleneck blocks, this is the bottleneck width;
            for basic blocks, this is the output channels.
        num_blocks (int): Number of residual blocks in this stage.
        downsample (bool): Whether to downsample at the start of this stage.
        stride (int): Stride for the first block in this stage, used for downsampling.
        expansion (int): Expansion factor for bottleneck blocks.
    """
    def __init__(self, block_cls: Type[nn.Module], mid_channels: int, num_blocks: int,
                 downsample: bool, stride: int = 2, expansion: int = 4):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            block_stride = stride if downsample and i == 0 else 1
            # First block in each stage needs projection (channel change or downsampling)
            use_proj = (i == 0)
            if issubclass(block_cls, ResidualBottleneckBlock):
                b = block_cls(mid_channels, block_stride, expansion, use_proj)
            elif issubclass(block_cls, ResidualBasicBlock):
                b = block_cls(mid_channels, block_stride, use_proj)
            else:
                raise ValueError("Unsupported block class")
            blocks.append(b)
        self.stage = nn.Sequential(*blocks)
        
    def forward(self, X: Tensor) -> Tensor:
        return self.stage(X)

class ResNet(nn.Module):
    """
    A ResNet model.
    
    Args:
        stem_channels (int): Number of output channels from the stem.
        block_cls (Type[nn.Module]): The residual block class to use.
        stages (List[Tuple[int, int]]): A list where each tuple contains
            (number of blocks, mid_channels) for each stage. mid_channels is the
            number of channels in the residual blocks before expansion.
            Example for ResNet-18: [(2, 64), (2, 128), (2, 256), (2, 512)]
        num_classes (int): Number of output classes.
    """
    def __init__(self, stem_channels: int, block_cls: Type[nn.Module],
                 stages: List[Tuple[int, int]], num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            self.stem(stem_channels),
        )
        for i, (num_blocks, mid_channels) in enumerate(stages):
            stride = 2 if i != 0 else 1 # Downsample except for first stage
            self.net.add_module(
                f'stage{i+1}',
                ResNetStage(
                    block_cls,
                    mid_channels=mid_channels,
                    num_blocks=num_blocks,
                    downsample=(i != 0),
                    stride=stride,
                )
            )
        self.net.add_module('head', self.head(num_classes))
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
    
    def stem(self, num_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    
    def head(self, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(), # Account for the last block's pre-activation
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes),
        )
        
class ResNet18(ResNet):
    def __init__(self, num_classes: int = 10):
        stages = [(2, 64), (2, 128), (2, 256), (2, 512)]
        super().__init__(stem_channels=64,
                         block_cls=ResidualBasicBlock,
                         stages=stages,
                         num_classes=num_classes)
        
class ResNet50(ResNet):
    def __init__(self, num_classes: int = 10):
        stages = [(3, 64), (4, 128), (6, 256), (3, 512)]
        super().__init__(stem_channels=64,
                         block_cls=ResidualBottleneckBlock,
                         stages=stages,
                         num_classes=num_classes)
    
if __name__ == "__main__":
    batch_size = 128
    num_epochs = 20 # 10 is mostly enough, 20 used for experiments
    lr = 0.05
    
    model = ResNet18(num_classes=10)
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
        log_path='logs/resnet_pre_act_experiment.json',
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
        save_path='models/resnet_pre_act.pt',
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
    