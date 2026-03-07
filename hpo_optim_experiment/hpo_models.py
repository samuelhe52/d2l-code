"""Model registry and objective builders for classification HPO scripts."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pruning_trainer import PruningTrainer
from utils.data import fashion_mnist
from utils.training import TrainingConfig

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attention_transformer.ViT import ViT
from modern_cnns.densenet import DenseNet18
from modern_cnns.resnet_pre_act import ResNet18


@dataclass(frozen=True)
class HpoModelSpec:
    key: str
    description: str
    default_study_name: str
    default_storage_file: str
    default_resize: int
    default_num_epochs: int
    default_batch_size_min: int
    default_batch_size_max: int
    default_batch_size_step: int
    objective_builder: Callable[[object, torch.device, dict[str, int | bool | None]], Callable[[optuna.Trial], float]]


def build_loader_pair(
    batch_size: int,
    resize: int,
    loader_kwargs: dict[str, int | bool | None],
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    data_root = ROOT / "data"
    train_loader = fashion_mnist(
        batch_size,
        resize=resize,
        data_root=str(data_root),
        **loader_kwargs,
    )
    val_loader = fashion_mnist(
        batch_size,
        train=False,
        resize=resize,
        data_root=str(data_root),
        **loader_kwargs,
    )
    return train_loader, val_loader


def init_lazy_model(model: nn.Module, resize: int, init_weights: Callable[[nn.Module], None]) -> None:
    model(torch.randn(1, 1, resize, resize))
    model.apply(init_weights)


def configure_trial(
    trial: optuna.Trial,
    args: object,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
) -> None:
    seed = args.seed + trial.number
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    trial.set_user_attr("pid", os.getpid())
    trial.set_user_attr("worker_index", args.worker_index)
    trial.set_user_attr("model", args.model)
    trial.set_user_attr("device", str(device))
    trial.set_user_attr("num_workers", args.num_workers)
    trial.set_user_attr("pin_memory", loader_kwargs["pin_memory"])


def run_classification_trial(
    *,
    trial: optuna.Trial,
    args: object,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
    resize: int,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    lr: float,
) -> float:
    dataloader, val_dataloader = build_loader_pair(
        batch_size=trial.params["batch_size"],
        resize=resize,
        loader_kwargs=loader_kwargs,
    )
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        lr=lr,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        verbose=args.verbose_trainer,
        silence=not args.verbose_trainer,
    )
    trainer = PruningTrainer(
        model,
        dataloader,
        val_dataloader,
        config,
        trial=trial,
        silence=not args.verbose_trainer,
    )

    try:
        trainer.train()
        return trainer.validate().get("val_acc", 0.0)
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            trial.set_user_attr("failure", "cuda_oom")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM") from exc
        raise


def build_densenet_objective(
    args: object,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        configure_trial(trial, args, device, loader_kwargs)
        batch_size = trial.suggest_int(
            "batch_size",
            args.batch_size_min,
            args.batch_size_max,
            step=args.batch_size_step,
        )
        cosine_annealing_t_max = trial.suggest_int("cosine_annealing_t_max", 15, 20)
        lr = trial.suggest_float("lr", 1e-3, 5e-1, log=True)
        momentum = 0.9

        model = DenseNet18(num_classes=10)
        init_fn = torch.nn.init.kaiming_uniform_

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_fn(module.weight)

        init_lazy_model(model, args.resize, init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_annealing_t_max)
        trial.set_user_attr("batch_size", batch_size)
        return run_classification_trial(
            trial=trial,
            args=args,
            device=device,
            loader_kwargs=loader_kwargs,
            resize=args.resize,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=lr,
        )

    return objective


def build_resnet_objective(
    args: object,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        configure_trial(trial, args, device, loader_kwargs)
        batch_size = trial.suggest_int(
            "batch_size",
            args.batch_size_min,
            args.batch_size_max,
            step=args.batch_size_step,
        )
        cosine_annealing_t_max = trial.suggest_int("cosine_annealing_t_max", 15, 20)
        lr = trial.suggest_float("lr", 1e-3, 5e-1, log=True)
        momentum = 0.9

        model = ResNet18(num_classes=10)
        init_fn = torch.nn.init.kaiming_uniform_

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_fn(module.weight)

        init_lazy_model(model, args.resize, init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_annealing_t_max)
        trial.set_user_attr("batch_size", batch_size)
        return run_classification_trial(
            trial=trial,
            args=args,
            device=device,
            loader_kwargs=loader_kwargs,
            resize=args.resize,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=lr,
        )

    return objective


def build_vit_objective(
    args: object,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        configure_trial(trial, args, device, loader_kwargs)
        batch_size = trial.suggest_int(
            "batch_size",
            args.batch_size_min,
            args.batch_size_max,
            step=args.batch_size_step,
        )
        embed_dim = trial.suggest_categorical("embed_dim", [256, 512])
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        num_blocks = trial.suggest_int("num_blocks", 2, 4)
        embed_dropout = trial.suggest_float("embed_dropout", 0.0, 0.3)
        blk_dropout = trial.suggest_float("blk_dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        cosine_annealing_t_max = trial.suggest_int("cosine_annealing_t_max", 10, 20)

        model = ViT(
            img_size=args.resize,
            patch_size=16,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=embed_dim * 4,
            num_blocks=num_blocks,
            num_classes=10,
            embed_dropout=embed_dropout,
            blk_dropout=blk_dropout,
        )

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        init_lazy_model(model, args.resize, init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_annealing_t_max)
        trial.set_user_attr("batch_size", batch_size)
        return run_classification_trial(
            trial=trial,
            args=args,
            device=device,
            loader_kwargs=loader_kwargs,
            resize=args.resize,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=lr,
        )

    return objective


MODEL_SPECS = {
    "densenet": HpoModelSpec(
        key="densenet",
        description="DenseNet Fashion-MNIST HPO worker",
        default_study_name="densenet-fashion-mnist",
        default_storage_file="densenet_optim.db",
        default_resize=96,
        default_num_epochs=20,
        default_batch_size_min=64,
        default_batch_size_max=512,
        default_batch_size_step=64,
        objective_builder=build_densenet_objective,
    ),
    "resnet": HpoModelSpec(
        key="resnet",
        description="ResNet Fashion-MNIST HPO worker",
        default_study_name="resnet-fashion-mnist",
        default_storage_file="resnet_optim.db",
        default_resize=96,
        default_num_epochs=20,
        default_batch_size_min=64,
        default_batch_size_max=512,
        default_batch_size_step=64,
        objective_builder=build_resnet_objective,
    ),
    "vit": HpoModelSpec(
        key="vit",
        description="ViT Fashion-MNIST HPO worker",
        default_study_name="vit-fashion-mnist",
        default_storage_file="vit_optim.db",
        default_resize=96,
        default_num_epochs=20,
        default_batch_size_min=32,
        default_batch_size_max=128,
        default_batch_size_step=32,
        objective_builder=build_vit_objective,
    ),
}