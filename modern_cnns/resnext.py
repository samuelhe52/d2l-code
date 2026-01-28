# This implementation of ResNeXt uses pre-activation residual blocks.

import torch
from torch import nn, Tensor
from utils.classfication import train, fashion_mnist
from utils import TrainingLogger
from utils.training_config import TrainingConfig
from typing import Dict, Any, Type, List, Tuple

class ResidualBlock(nn.Module):
    """
    A bottleneck residual block for ResNeXt.
    
    Args:
        mid_channels (int): Number of channels in the grouped convolution.
        cardinality (int): Number of groups in the grouped convolution.
        stride (int): Stride for the second convolutional layer.
        expansion (int): Channel expansion factor applied after the third conv.
        use_proj (bool): Whether to use a projection shortcut.
    """
    def __init__(self, mid_channels: int, cardinality: int,
                 stride: int = 1, expansion: int = 4, use_proj: bool = True):
        super().__init__()
        out_channels = mid_channels * expansion
        conv1 = nn.LazyConv2d(mid_channels, kernel_size=1, bias=False)
        conv2 = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(mid_channels, kernel_size=3, stride=stride,
                          padding=1, bias=False, groups=cardinality),
        )
        conv3 = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
        )
        self.norm_act = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.use_proj = use_proj
        self.proj = nn.LazyConv2d(out_channels, kernel_size=1,
                                  stride=stride, bias=False) if use_proj else None
        self.conv_layers = nn.Sequential(conv1, conv2, conv3)

    def forward(self, X: Tensor) -> Tensor:
        if self.use_proj:
            # Apply BN/ReLU to both branches
            X = self.norm_act(X)
            out = self.conv_layers(X)
            shortcut = self.proj(X)
        else:
            # Identity shortcut: BN/ReLU only on conv branch
            out = self.conv_layers(self.norm_act(X))
            shortcut = X
        return out + shortcut

class ResNeXtStage(nn.Module):
    """
    A stage consisting of multiple residual blocks.
    
    Args:
        mid_channels (int): Number of channels in the grouped convolutions.
        cardinality (int): Number of groups in the grouped convolution.
        num_blocks (int): Number of residual blocks in this stage.
        downsample (bool): Whether to downsample at the start of this stage.
        stride (int): Stride for the first block in this stage, used for downsampling.
        expansion (int): Expansion factor for bottleneck blocks.
    """
    def __init__(self, mid_channels: int, cardinality: int,
                 num_blocks: int, downsample: bool, stride: int = 2, expansion: int = 4):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            block_stride = stride if downsample and i == 0 else 1
            # First block in each stage needs projection (channel expansion or downsampling)
            # Subsequent blocks have matching dimensions
            use_proj = (i == 0)
            blocks.append(
                ResidualBlock(
                    mid_channels=mid_channels,
                    cardinality=cardinality,
                    stride=block_stride,
                    expansion=expansion,
                    use_proj=use_proj)
                )
        self.stage = nn.Sequential(*blocks)
        
    def forward(self, X: Tensor) -> Tensor:
        return self.stage(X)

class ResNeXt(nn.Module):
    """
    A ResNeXt model with bottleneck blocks and grouped convolutions.
    
    Args:
        stem_channels (int): Number of output channels from the stem.
        cardinality (int): Number of groups in the grouped convolution.
        stages (List[Tuple[int, int]]): A list where each tuple contains
            (number of blocks, mid_channels) for each stage. mid_channels is the
            number of channels in the grouped convolutions (cardinality * width_per_group).
            Example for ResNeXt-50 (32x4d): [(3, 128), (4, 256), (6, 512), (3, 1024)]
        expansion (int): Channel expansion factor for bottleneck blocks.
        num_classes (int): Number of output classes.
    """
    def __init__(self, stem_channels: int, cardinality: int,
                 stages: List[Tuple[int, int]], expansion: int = 4,
                 num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(self._stem(stem_channels))
        
        for i, (num_blocks, mid_channels) in enumerate(stages):
            stride = 2 if i != 0 else 1  # Downsample except for first stage
            self.net.add_module(
                f'stage{i+1}',
                ResNeXtStage(
                    mid_channels=mid_channels,
                    cardinality=cardinality,
                    num_blocks=num_blocks,
                    downsample=(i != 0),
                    stride=stride,
                    expansion=expansion,
                )
            )
        self.net.add_module('head', self._head(num_classes))
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
    
    def _stem(self, num_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    
    def _head(self, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),  # Account for the last block's pre-activation
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes),
        )
        
class ResNeXt50_32x4d(ResNeXt):
    """ResNeXt-50 with cardinality=32 and base width=4 (32x4d)."""
    def __init__(self, num_classes: int = 10):
        # mid_channels = cardinality * width_per_group = 32 * 4 = 128 for first stage
        # Then doubles each stage: 128 -> 256 -> 512 -> 1024
        # expansion=2 gives outputs: 256 -> 512 -> 1024 -> 2048
        stages = [(3, 128), (4, 256), (6, 512), (3, 1024)]
        super().__init__(stem_channels=64,
                         cardinality=32,
                         stages=stages,
                         expansion=2,
                         num_classes=num_classes)
        
class ResNeXt101_32x8d(ResNeXt):
    """ResNeXt-101 with cardinality=32 and base width=8 (32x8d)."""
    def __init__(self, num_classes: int = 10):
        # mid_channels = cardinality * width_per_group = 32 * 8 = 256 for first stage
        # Then doubles each stage: 256 -> 512 -> 1024 -> 2048
        # expansion=2 gives outputs: 512 -> 1024 -> 2048 -> 4096
        stages = [(3, 256), (4, 512), (23, 1024), (3, 2048)]
        super().__init__(stem_channels=64,
                         cardinality=32,
                         stages=stages,
                         expansion=2,
                         num_classes=num_classes)
    
if __name__ == "__main__":
    batch_size = 128
    num_epochs = 20 # 10 is mostly enough, 20 used for experiments
    lr = 0.05
    
    model = ResNeXt50_32x4d(num_classes=10)
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
        log_path='logs/resnext_experiment.json',
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
        save_path='models/resnext.pt',
        logger=logger,
    )

    train(
        model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )
    
    logger.summary()
    logger.save()
    