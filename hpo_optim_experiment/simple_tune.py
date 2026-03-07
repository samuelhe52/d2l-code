"""Minimal Ray Tune helpers for experimentation scripts."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import ray
import torch
from ray import tune
from ray.tune import ResultGrid, RunConfig
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.data import fashion_mnist
from utils.training import TrainingConfig
from utils.training.classification import ClassificationTrainer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class TuneReportTrainer(ClassificationTrainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("silence", True)
        super().__init__(*args, **kwargs)

    def validate(self) -> dict[str, float]:
        result = super().validate()
        val_acc = result.get("val_acc", 0.0)
        val_loss = result.get("val_loss", 0.0)
        if not self.cfg.silence:
            print(
                f"Evaluated val_acc: {val_acc:.4f} at epoch {self.current_epoch}. "
                f"Process ID: {os.getpid()}"
            )
        tune.report(
            {
                "val_acc": val_acc,
                "val_loss": val_loss,
                "epoch": self.current_epoch,
            }
        )
        return result


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available")
    return torch.device(device_name)


def prepare_storage(storage_path: str | Path) -> str:
    path = Path(storage_path)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_loader_kwargs(
    device: torch.device,
    *,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> dict[str, int | bool | None]:
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory if pin_memory is not None else device.type == "cuda",
        "persistent_workers": (
            persistent_workers if persistent_workers is not None else num_workers > 0
        ),
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }


def fashion_mnist_loaders(
    batch_size: int,
    *,
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


def init_lazy_model(model: nn.Module, resize: int, init_weights: callable) -> None:
    model(torch.randn(1, 1, resize, resize))
    model.apply(init_weights)


def seed_from_config(config: dict[str, Any], base_seed: int) -> int:
    digest = hashlib.sha256(repr(sorted(config.items())).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def build_trial_resources(
    device: torch.device,
    *,
    cpus_per_trial: float,
    cuda_virtual_jobs: int,
) -> dict[str, float]:
    resources = {"cpu": float(cpus_per_trial)}
    if device.type == "cuda":
        if cuda_virtual_jobs < 1:
            raise ValueError("cuda_virtual_jobs must be at least 1")
        resources["gpu"] = 1.0 / float(cuda_virtual_jobs)
    return resources


def init_ray_runtime(
    *,
    ray_address: str | None,
    ray_num_cpus: int | None,
    ray_num_gpus: float | None,
    show_progress: bool,
) -> None:
    if ray.is_initialized():
        return
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "log_to_driver": show_progress,
    }
    if ray_address:
        init_kwargs["address"] = ray_address
    else:
        if ray_num_cpus is not None:
            init_kwargs["num_cpus"] = ray_num_cpus
        if ray_num_gpus is not None:
            init_kwargs["num_gpus"] = ray_num_gpus
    ray.init(**init_kwargs)


def print_best_result(results: ResultGrid, *, metric: str, mode: str) -> None:
    best_result = results.get_best_result(metric=metric, mode=mode)
    print("Best trial:")
    print(f"  Value: {best_result.metrics.get(metric)}")
    print("  Params:")
    for key, value in best_result.config.items():
        print(f"    {key}: {value}")


def run_tune_experiment(
    *,
    experiment_name: str,
    search_space: dict[str, Any],
    trainable: callable,
    metric: str,
    mode: str,
    num_samples: int,
    num_epochs: int,
    storage_path: str | Path,
    device: torch.device,
    cpus_per_trial: float,
    cuda_virtual_jobs: int,
    grace_period: int = 3,
    reduction_factor: int = 3,
    max_concurrent_trials: int | None = None,
    ray_address: str | None = None,
    ray_num_cpus: int | None = None,
    ray_num_gpus: float | None = None,
    show_progress: bool = True,
) -> ResultGrid:
    init_ray_runtime(
        ray_address=ray_address,
        ray_num_cpus=ray_num_cpus,
        ray_num_gpus=ray_num_gpus,
        show_progress=show_progress,
    )
    resources = build_trial_resources(
        device,
        cpus_per_trial=cpus_per_trial,
        cuda_virtual_jobs=cuda_virtual_jobs,
    )
    effective_grace_period = max(1, min(grace_period, num_epochs))
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=num_epochs,
        grace_period=effective_grace_period,
        reduction_factor=reduction_factor,
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=prepare_storage(storage_path),
            verbose=1 if show_progress else 0,
        ),
    )
    results = tuner.fit()
    print_best_result(results, metric=metric, mode=mode)
    return results


def run_classification_trial(
    *,
    config: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    lr: float,
    resize: int,
    num_epochs: int,
    device: torch.device,
    loader_kwargs: dict[str, int | bool | None],
    seed: int = 42,
    verbose: bool = False,
    torch_num_threads: int = 1,
    torch_num_interop_threads: int = 1,
) -> dict[str, float]:
    trial_seed = seed_from_config(config, seed)
    torch.manual_seed(trial_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(trial_seed)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, torch_num_threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, torch_num_interop_threads))

    train_loader, val_loader = fashion_mnist_loaders(
        int(config["batch_size"]),
        resize=resize,
        loader_kwargs=loader_kwargs,
    )
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        verbose=verbose,
        silence=not verbose,
    )
    trainer = TuneReportTrainer(
        model,
        train_loader,
        val_loader,
        training_config,
        silence=not verbose,
    )

    try:
        final_metrics = trainer.train() or {}
        return {
            "val_acc": final_metrics.get("val_acc", 0.0),
            "val_loss": final_metrics.get("val_loss", 0.0),
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            tune.report({"val_acc": 0.0, "val_loss": 1.0, "epoch": 0, "oom": 1.0})
            return {"val_acc": 0.0, "val_loss": 1.0}
        raise