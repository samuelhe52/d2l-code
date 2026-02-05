import torch
from torch import nn, Tensor
from typing import Any, Dict
from utils.training import ClassificationTrainer, TrainingLogger, TrainingConfig
from utils.data import fashion_mnist

class LeNetModern(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)

if __name__ == "__main__":
    batch_size = 256
    num_epochs = 20
    lr = 0.05
    weight_decay = 0.0
    
    # Select device: prefer CUDA, then MPS (Apple), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = LeNetModern().to(device)
    dataloader = fashion_mnist(batch_size, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, data_root='data/')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    init_fn = torch.nn.init.kaiming_uniform_

    hparams: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'init_fn': init_fn.__name__,
        'pooling': 'MaxPool2d',
    }
    
    # Set up logger with all hyperparameters
    logger = TrainingLogger(
        log_path='logs/lenet_experiment.json',
        hparams=hparams
    )

    # Initialize model parameters
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    model.apply(init_weights)
    
    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        logger=logger,
        save_path='models/lenet.pt',
        device=device,
    )
    
    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()

    logger.summary()
    logger.save()
    