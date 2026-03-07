import sys
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.data import fashion_mnist
from utils.training import ClassificationTrainer, TrainingConfig

# Path hack to allow importing from attention_transformer
sys.path.append(str(Path(__file__).parent.parent))
from attention_transformer.ViT import ViT


def objective(trial: optuna.Trial) -> float:
    img_size = 96
    patch_size = 16
    batch_size = trial.suggest_int("batch_size", 32, 128)
    num_epochs = 20
    embed_dim = trial.suggest_categorical("embed_dim", [256, 512])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_blocks = trial.suggest_int("num_blocks", 2, 4)
    embed_dropout = trial.suggest_float("embed_dropout", 0.0, 0.3)
    blk_dropout = trial.suggest_float("blk_dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    cosine_annealing_t_max = trial.suggest_int("cosine_annealing_t_max", 10, 20)
    mlp_hidden_dim = embed_dim * 4

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        num_blocks=num_blocks,
        num_classes=10,
        embed_dropout=embed_dropout,
        blk_dropout=blk_dropout,
    )
    dataloader = fashion_mnist(batch_size, resize=img_size, data_root="data/")
    val_dataloader = fashion_mnist(batch_size, train=False, resize=img_size, data_root="data/")

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    model(torch.randn(1, 1, img_size, img_size))
    model.apply(init_weights)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optim, T_max=cosine_annealing_t_max)

    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
    )

    trainer = ClassificationTrainer(model, dataloader, val_dataloader, config)
    trainer.train()

    return trainer.validate().get("val_acc", 0.0)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")