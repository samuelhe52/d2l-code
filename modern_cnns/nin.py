import torch
from torch import nn, Tensor
from utils.data import fashion_mnist
from utils.training import ClassificationTrainer, TrainingLogger, TrainingConfig
from typing import Dict, Any

def nin_block(num_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LazyConv2d(num_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1), nn.ReLU()
    )

class NiN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)

if __name__ == "__main__":
    batch_size = 256
    num_epochs = 15
    lr = 0.05
    
    model = NiN(num_classes=10)
    dataloader = fashion_mnist(batch_size, resize=84, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, resize=84, data_root='data/')
    init_fn = torch.nn.init.kaiming_uniform_
    
    hparams: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'init_fn': init_fn.__name__,
    }
    
    logger = TrainingLogger(
        log_path='logs/nin_experiment.json',
        hparams=hparams
    )
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init_fn(m.weight)
    
    model.forward(torch.randn(1, 1, 84, 84))  # Initialize lazy layers
    model.apply(init_weights)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        save_path='models/nin.pt',
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
    