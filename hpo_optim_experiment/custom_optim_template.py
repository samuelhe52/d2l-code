"""Copy this file when you want a quick custom Ray Tune experiment."""

from __future__ import annotations

try:
    from .simple_tune import resolve_device, run_tune_experiment
except ImportError:
    from simple_tune import resolve_device, run_tune_experiment

from ray import tune

EXPERIMENT_NAME = "custom-experiment"
DEVICE_NAME = "auto"
NUM_SAMPLES = 20
NUM_EPOCHS = 10
CPUS_PER_TRIAL = 2.0
CUDA_VIRTUAL_JOBS = 4
MAX_CONCURRENT_TRIALS = None
STORAGE_PATH = "logs/ray_results"
RAY_ADDRESS = None
RAY_NUM_CPUS = None
RAY_NUM_GPUS = None
SHOW_PROGRESS = True

SEARCH_SPACE = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "hidden_dim": tune.choice([128, 256, 512]),
}


def trainable(config: dict[str, float]) -> None:
    raise NotImplementedError("Replace this with your own trainable")


def main() -> None:
    device = resolve_device(DEVICE_NAME)
    run_tune_experiment(
        experiment_name=EXPERIMENT_NAME,
        search_space=SEARCH_SPACE,
        trainable=trainable,
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