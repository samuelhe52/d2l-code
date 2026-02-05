import torch
from torch import nn, Tensor
from typing import List, Tuple, Any, Dict
from utils.data import fashion_mnist
from utils.training import ClassificationTrainer
from utils import TrainingLogger
from utils import TrainingConfig

def vgg_block(num_convs: int, out_channels: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, arch: List[Tuple[int, int]], num_classes: int = 10):
        super().__init__()
        vgg_blks = []
        for (num_convs, out_channels) in arch:
            vgg_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *vgg_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 10
    lr = 0.01
    
    vgg11_arch: List[Tuple[int, int]] = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
        
    model = VGG(arch=vgg11_arch)
    dataloader = fashion_mnist(batch_size, resize=224, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, resize=224, data_root='data/')
    init_fn = torch.nn.init.kaiming_uniform_
    
    hparams: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'init_fn': init_fn.__name__,
    }
    
    logger = TrainingLogger(
        log_path='logs/vgg_experiment.json',
        hparams=hparams
    )
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init_fn(m.weight)
    
    model.forward(torch.randn(1, 1, 224, 224))  # Initialize lazy layers
    model.apply(init_weights)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        save_path='models/vgg.pt',
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
    