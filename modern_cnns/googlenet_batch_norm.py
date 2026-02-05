import torch
from torch import nn, Tensor
from utils.classfication import fashion_mnist
from utils.training import ClassificationTrainer
from utils import TrainingLogger
from utils import TrainingConfig
from typing import Dict, Any

class Inception(nn.Module):
    """
    Inception block as described in the GoogLeNet paper.
    
    Args:
        c1: Number of output channels for the 1x1 conv.
        c2: Tuple with number of output channels for the 1x1 and 3x3 convs in the 3x3 branch.
        c3: Tuple with number of output channels for the 1x1 and 5x5 convs in the 5x5 branch.
        c4: Number of output channels for the max-pooling branch.
    """
    def __init__(self, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.LazyConv2d(c1, kernel_size=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(c2[0], kernel_size=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(c2[1], kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(c3[0], kernel_size=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # Replace 5x5 conv with two 3x3 convs
            nn.LazyConv2d(c3[1], kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(c3[1], kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(c4, kernel_size=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )

    def forward(self, X: Tensor) -> Tensor:
        return torch.cat(
            [self.branch1(X), self.branch2(X), self.branch3(X), self.branch4(X)],
            dim=1,
        )

class GoogleNetBN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            self.stem(),
            self.inception1(),
            self.inception2(),
            self.inception3(),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
    
    def stem(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LazyConv2d(64, kernel_size=1, bias=False),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
    def inception1(self) -> nn.Sequential:
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
    def inception2(self) -> nn.Sequential:
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
    def inception3(self) -> nn.Sequential:
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
if __name__ == "__main__":
    batch_size = 128
    num_epochs = 10
    lr = 0.05
    
    model = GoogleNetBN(num_classes=10)
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
        log_path='logs/googlenet_bn_experiment.json',
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
        save_path='models/googlenet_bn.pt',
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
    