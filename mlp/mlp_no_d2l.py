import torch
from torch import nn, Tensor
from typing import Any, Dict
from utils import TrainingLogger, TrainingConfig
from utils.training import ClassificationTrainer
from utils.data import fashion_mnist


class MLP(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int,
                 num_hiddens_1: int, num_hiddens_2: int,
                 dropout_1: float = 0.0, dropout_2: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(num_inputs, num_hiddens_1),
            nn.ReLU(), nn.Dropout(dropout_1),
            nn.Linear(num_hiddens_1, num_hiddens_2),
            nn.ReLU(), nn.Dropout(dropout_2),
            nn.Linear(num_hiddens_2, num_outputs)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)

if __name__ == "__main__":
    hparams: Dict[str, Any] = {
        'num_inputs': 28 * 28,
        'num_outputs': 10,
        'num_hiddens_1': 512,
        'num_hiddens_2': 256,
        'dropout_1': 0.40,
        'dropout_2': 0.40,
        # 'dropout_1': 0.00,
        # 'dropout_2': 0.00,
    }

    batch_size = 256
    num_epochs = 20
    lr = 0.1
    # weight_decay = 5e-4
    weight_decay = 0.0

    # Select device: prefer CUDA, then MPS (Apple), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MLP(**hparams).to(device)
    dataloader = fashion_mnist(batch_size, data_root='data/')
    val_dataloader = fashion_mnist(batch_size, train=False, data_root='data/')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up logger with all hyperparameters
    logger = TrainingLogger(
        log_path='logs/mlp_experiment.json',
        hparams={
            **hparams,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
        }
    )
    
    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        logger=logger,
        device=device,
    )
    
    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()
    
    logger.summary()
    logger.save()
