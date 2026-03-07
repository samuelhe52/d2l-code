"""Launch multiple HPO worker processes against one Optuna study."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import optuna

from hpo_worker import apply_model_defaults, build_parser, prepare_storage

WORKER_SCRIPT = Path(__file__).resolve().parent / "hpo_worker.py"


def build_launcher_parser(
    default_model: str | None = None,
    *,
    include_model: bool = True,
    description: str = "Generic HPO multi-process launcher",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        parents=[
            build_parser(
                default_model=default_model,
                add_help=False,
                include_worker_index=False,
                include_model=include_model,
                description=description,
            )
        ],
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    return parser


def parse_launcher_args(
    default_model: str | None = None,
    *,
    include_model: bool = True,
    description: str = "Generic HPO multi-process launcher",
) -> argparse.Namespace:
    parser = build_launcher_parser(
        default_model=default_model,
        include_model=include_model,
        description=description,
    )
    args = parser.parse_args()
    if not include_model:
        args.model = default_model or "densenet"
    args.worker_index = 0
    return apply_model_defaults(args)


def build_worker_command(args: argparse.Namespace, worker_index: int) -> list[str]:
    command = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--model",
        args.model,
        "--study-name",
        args.study_name,
        "--storage",
        args.storage,
        "--n-trials",
        str(args.n_trials),
        "--seed",
        str(args.seed),
        "--worker-index",
        str(worker_index),
        "--device",
        args.device,
        "--num-epochs",
        str(args.num_epochs),
        "--resize",
        str(args.resize),
        "--batch-size-min",
        str(args.batch_size_min),
        "--batch-size-max",
        str(args.batch_size_max),
        "--batch-size-step",
        str(args.batch_size_step),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--torch-num-interop-threads",
        str(args.torch_num_interop_threads),
        "--pruner-min-resource",
        str(args.pruner_min_resource),
        "--pruner-reduction-factor",
        str(args.pruner_reduction_factor),
    ]
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.pin_memory is not None:
        command.append("--pin-memory" if args.pin_memory else "--no-pin-memory")
    if args.persistent_workers is not None:
        command.append(
            "--persistent-workers" if args.persistent_workers else "--no-persistent-workers"
        )
    if args.verbose_trainer:
        command.append("--verbose-trainer")
    if args.show_progress:
        command.append("--show-progress")
    return command


def print_best_trial(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def main(
    default_model: str | None = None,
    *,
    include_model: bool = True,
    description: str = "Generic HPO multi-process launcher",
) -> None:
    args = parse_launcher_args(
        default_model=default_model,
        include_model=include_model,
        description=description,
    )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    args.storage = prepare_storage(args.storage)
    processes: list[subprocess.Popen[bytes]] = []
    for worker_index in range(args.workers):
        command = build_worker_command(args, worker_index)
        print(f"Launching worker {worker_index}: {' '.join(command)}")
        processes.append(subprocess.Popen(command, cwd=WORKER_SCRIPT.parent.parent))

    exit_code = 0
    for worker_index, process in enumerate(processes):
        worker_exit_code = process.wait()
        if worker_exit_code != 0 and exit_code == 0:
            exit_code = worker_exit_code
        print(f"Worker {worker_index} exited with code {worker_exit_code}")

    if exit_code != 0:
        raise SystemExit(exit_code)

    print_best_trial(args)


if __name__ == "__main__":
    main()