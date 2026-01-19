# This is an improved implementation of vgg.py
# with adaptations targeting better performance on Fashion-MNIST.

import torch
from torch import nn
from utils.classfication import train, fashion_mnist
from utils import TrainingLogger
from utils.training_config import TrainingConfig

def vgg_block(num_convs, out_channels, pool = True):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        vgg_blks = []
        for (num_convs, out_channels, pool) in arch:
            vgg_blks.append(vgg_block(num_convs, out_channels, pool))
        self.net = nn.Sequential(
            *vgg_blks,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(), nn.Dropout(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X):
        return self.net(X)

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 10
    lr = 0.01
    
    vgg11_arch = [(1, 64, False), (1, 128, False), (2, 256, True), (2, 512, True), (2, 512, True)]
        
    model = VGG(arch=vgg11_arch)
    dataloader = fashion_mnist(batch_size, resize=84, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, resize=84, data_root='data/')
    init_fn = torch.nn.init.kaiming_uniform_
    
    hparams = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'init_fn': init_fn.__name__,
    }
    
    logger = TrainingLogger(
        log_path='logs/vgg_improved_experiment.json',
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
        save_path='models/vgg_improved.pt',
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
    