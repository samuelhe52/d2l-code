# This file uses modern_cnns/resnet_pre_act.py as a base.
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from utils.data import fashion_mnist
from utils.training import (
    TrainingConfig,
)
from pathlib import Path
import sys
from pruning_trainer import PruningTrainer

# Path hacks to allow importing from modern_cnns
sys.path.append(str(Path(__file__).parent.parent))
from modern_cnns.resnet_pre_act import ResNet18

def objective(trial: optuna.Trial) -> float:
    batch_size = trial.suggest_int("batch_size", 64, 256)
    num_epochs = 20
    cosine_annealing_T_max = trial.suggest_int("cosine_annealing_T_max", 15, 20)
    lr = trial.suggest_float("lr", 1e-3, 5e-1, log=True)
    momentum = 0.9
    
    model = ResNet18(num_classes=10)
    dataloader = fashion_mnist(batch_size, resize=96, data_root="data/")
    val_dataloader = fashion_mnist(batch_size, train=False, resize=96, data_root="data/")
    init_fn = torch.nn.init.kaiming_uniform_

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            init_fn(module.weight)

    model.forward(torch.randn(1, 1, 96, 96))  # Initialize lazy layers
    model.apply(init_weights)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = CosineAnnealingLR(optim, T_max=cosine_annealing_T_max)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
    )

    trainer = PruningTrainer(
        model,
        dataloader,
        val_dataloader,
        config,
        trial=trial,
        silence=True,
    )
    trainer.train()
    
    return trainer.validate().get('val_acc', 0.0)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


