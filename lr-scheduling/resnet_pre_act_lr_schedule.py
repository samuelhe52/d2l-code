# This is an adaptation of /modern_cnns/resnet_pre_act.py, adding
# lr scheduling and momentum.

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.data import fashion_mnist
from utils.training import (
    ClassificationTrainer,
    TrainingLogger,
    TrainingConfig,
)
from typing import Dict, Any
from pathlib import Path
import sys

# Path hacks to allow importing from modern_cnns
sys.path.append(str(Path(__file__).parent.parent))
from modern_cnns.resnet_pre_act import ResNet18


if __name__ == "__main__":
    batch_size = 128
    num_epochs = 18  # 10 is mostly enough, 20 used for experiments
    lr = 0.05

    model = ResNet18(num_classes=10)
    dataloader = fashion_mnist(batch_size, resize=96, data_root="data/")
    val_dataloader = fashion_mnist(batch_size, train=False, resize=96, data_root="data/")
    init_fn = torch.nn.init.kaiming_uniform_

    hparams: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "init_fn": init_fn.__name__,
        "optimizer": "SGD",
        "momentum": 0.9,
        "scheduler": "CosineAnnealingLR",
    }

    logger = TrainingLogger(
        log_path="logs/resnet_pre_act_lr_schedule_experiment.json",
        hparams=hparams,
    )

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            init_fn(module.weight)

    model.forward(torch.randn(1, 1, 96, 96))  # Initialize lazy layers
    model.apply(init_weights)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = CosineAnnealingLR(optim, T_max=15)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        save_path="models/resnet_pre_act_lr_schedule.pt",
        logger=logger,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()

    logger.summary()
