"""ResNet Fashion-MNIST Ray Tune experiment script."""

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

from modern_cnns.resnet_pre_act import ResNet18

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

EXPERIMENT_NAME = "resnet-fashion-mnist"
DEVICE_NAME = "auto"
NUM_SAMPLES = 20
NUM_EPOCHS = 20
RESIZE = 96
NUM_WORKERS = 4
CUDA_VIRTUAL_JOBS = 4

if torch.cuda.is_available():
    CPUS_PER_TRIAL = 2.0
    MAX_CONCURRENT_TRIALS = None
else:
    CPUS_PER_TRIAL = 8.0
    MAX_CONCURRENT_TRIALS = 1

STORAGE_PATH = ROOT / "logs" / "ray_results"
RAY_ADDRESS = None
RAY_NUM_CPUS = None
RAY_NUM_GPUS = None
SHOW_PROGRESS = True
VERBOSE_TRAINER = False
TORCH_NUM_THREADS = 1
TORCH_NUM_INTEROP_THREADS = 1

SEARCH_SPACE = {
    "batch_size": tune.choice(list(range(64, 513, 64))),
    "cosine_annealing_t_max": tune.choice(list(range(15, 21))),
    "lr": tune.loguniform(1e-3, 5e-1),
}


def build_trainable(device: torch.device):
    loader_kwargs = build_loader_kwargs(device, num_workers=NUM_WORKERS)

    def trainable(config: dict[str, float]) -> dict[str, float]:
        model = ResNet18(num_classes=10)

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_uniform_(module.weight)

        init_lazy_model(model, RESIZE, init_weights)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config["lr"]),
            momentum=0.9,
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


