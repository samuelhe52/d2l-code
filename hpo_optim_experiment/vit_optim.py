"""ViT Fashion-MNIST Ray Tune experiment script."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ray import tune
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from attention_transformer.ViT import ViT

try:
    from .simple_tune import (
        build_loader_kwargs,
        init_lazy_model,
        resolve_device,
        run_classification_trial,
        run_tune_experiment,
    )
except ImportError:
    from simple_tune import (
        build_loader_kwargs,
        init_lazy_model,
        resolve_device,
        run_classification_trial,
        run_tune_experiment,
    )

EXPERIMENT_NAME = "vit-fashion-mnist"
DEVICE_NAME = "auto"
NUM_SAMPLES = 20
NUM_EPOCHS = 20
RESIZE = 96
NUM_WORKERS = 4
CPUS_PER_TRIAL = 2.0
CUDA_VIRTUAL_JOBS = 2
MAX_CONCURRENT_TRIALS = None
STORAGE_PATH = ROOT / "logs" / "ray_results"
RAY_ADDRESS = None
RAY_NUM_CPUS = None
RAY_NUM_GPUS = None
SHOW_PROGRESS = True
VERBOSE_TRAINER = False
TORCH_NUM_THREADS = 1
TORCH_NUM_INTEROP_THREADS = 1

SEARCH_SPACE = {
    "batch_size": tune.choice(list(range(32, 129, 32))),
    "embed_dim": tune.choice([256, 512]),
    "num_heads": tune.choice([4, 8]),
    "num_blocks": tune.choice([2, 3, 4]),
    "embed_dropout": tune.uniform(0.0, 0.3),
    "blk_dropout": tune.uniform(0.0, 0.3),
    "lr": tune.loguniform(1e-4, 5e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "cosine_annealing_t_max": tune.choice(list(range(10, 21))),
}


def build_trainable(device: torch.device):
    loader_kwargs = build_loader_kwargs(device, num_workers=NUM_WORKERS)

    def trainable(config: dict[str, float]) -> dict[str, float]:
        model = ViT(
            img_size=RESIZE,
            patch_size=16,
            embed_dim=int(config["embed_dim"]),
            num_heads=int(config["num_heads"]),
            mlp_hidden_dim=int(config["embed_dim"]) * 4,
            num_blocks=int(config["num_blocks"]),
            num_classes=10,
            embed_dropout=float(config["embed_dropout"]),
            blk_dropout=float(config["blk_dropout"]),
        )

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        init_lazy_model(model, RESIZE, init_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(config["cosine_annealing_t_max"]),
        )
        return run_classification_trial(
            config=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=float(config["lr"]),
            resize=RESIZE,
            num_epochs=NUM_EPOCHS,
            device=device,
            loader_kwargs=loader_kwargs,
            verbose=VERBOSE_TRAINER,
            torch_num_threads=TORCH_NUM_THREADS,
            torch_num_interop_threads=TORCH_NUM_INTEROP_THREADS,
        )

    return trainable


def main() -> None:
    device = resolve_device(DEVICE_NAME)
    run_tune_experiment(
        experiment_name=EXPERIMENT_NAME,
        search_space=SEARCH_SPACE,
        trainable=build_trainable(device),
        metric="val_acc",
        mode="max",
        num_samples=NUM_SAMPLES,
        num_epochs=NUM_EPOCHS,
        storage_path=STORAGE_PATH,
        device=device,
        cpus_per_trial=CPUS_PER_TRIAL,
        cuda_virtual_jobs=CUDA_VIRTUAL_JOBS,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        ray_address=RAY_ADDRESS,
        ray_num_cpus=RAY_NUM_CPUS,
        ray_num_gpus=RAY_NUM_GPUS,
        show_progress=SHOW_PROGRESS,
    )


if __name__ == "__main__":
    main()