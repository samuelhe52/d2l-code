"""Generic Optuna worker for model-specific HPO objectives."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import optuna
import torch

from hpo_models import MODEL_SPECS, ROOT


def build_parser(
    default_model: str | None = None,
    *,
    add_help: bool = True,
    include_worker_index: bool = True,
    include_model: bool = True,
    description: str = "Generic Fashion-MNIST HPO worker",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    if include_model:
        parser.add_argument(
            "--model",
            default=default_model or "densenet",
            choices=sorted(MODEL_SPECS),
            help="Model objective to optimize",
        )
    parser.add_argument("--study-name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL")
    parser.add_argument("--n-trials", type=int, default=20, help="Trials to run in this worker")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for the sampler")
    if include_worker_index:
        parser.add_argument("--worker-index", type=int, default=0, help="Worker index for metadata")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument("--num-epochs", type=int, default=None, help="Epochs per trial")
    parser.add_argument("--resize", type=int, default=None, help="Input image size")
    parser.add_argument("--batch-size-min", type=int, default=None, help="Minimum batch size")
    parser.add_argument("--batch-size-max", type=int, default=None, help="Maximum batch size")
    parser.add_argument("--batch-size-step", type=int, default=None, help="Batch size step")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes per HPO worker",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host memory for CUDA transfers",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive across epochs",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="Torch intra-op CPU threads per worker process",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=1,
        help="Torch inter-op CPU threads per worker process",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable Optuna progress bar for this worker",
    )
    parser.add_argument(
        "--verbose-trainer",
        action="store_true",
        help="Enable trainer progress bars and epoch logging",
    )
    parser.add_argument(
        "--pruner-min-resource",
        type=int,
        default=3,
        help="Minimum resource before pruning decisions",
    )
    parser.add_argument(
        "--pruner-reduction-factor",
        type=int,
        default=3,
        help="Successive halving reduction factor",
    )
    return parser


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    spec = MODEL_SPECS[args.model]
    if args.study_name is None:
        args.study_name = spec.default_study_name
    if args.storage is None:
        args.storage = f"sqlite:///{(ROOT / 'logs' / 'optuna' / spec.default_storage_file).as_posix()}"
    if args.resize is None:
        args.resize = spec.default_resize
    if args.num_epochs is None:
        args.num_epochs = spec.default_num_epochs
    if args.batch_size_min is None:
        args.batch_size_min = spec.default_batch_size_min
    if args.batch_size_max is None:
        args.batch_size_max = spec.default_batch_size_max
    if args.batch_size_step is None:
        args.batch_size_step = spec.default_batch_size_step
    return args


def parse_args(
    default_model: str | None = None,
    *,
    include_model: bool = True,
    description: str = "Generic Fashion-MNIST HPO worker",
) -> argparse.Namespace:
    parser = build_parser(
        default_model=default_model,
        include_model=include_model,
        description=description,
    )
    args = parser.parse_args()
    if not include_model:
        args.model = default_model or "densenet"
    return apply_model_defaults(args)


def resolve_device(device_name: str) -> torch.device:
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


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    torch.set_num_threads(max(1, args.torch_num_threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, args.torch_num_interop_threads))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def prepare_storage(storage_url: str) -> str:
    if not storage_url.startswith("sqlite:///"):
        return storage_url

    db_path = Path(storage_url.removeprefix("sqlite:///"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    return f"sqlite:///{db_path.as_posix()}"


def build_loader_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, int | bool | None]:
    pin_memory = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    persistent_workers = (
        args.persistent_workers if args.persistent_workers is not None else args.num_workers > 0
    )
    return {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
    }


def create_study(args: argparse.Namespace) -> optuna.study.Study:
    storage = prepare_storage(args.storage)
    args.storage = storage
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=args.pruner_min_resource,
        reduction_factor=args.pruner_reduction_factor,
    )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


def print_best_trial(study: optuna.study.Study) -> None:
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def run_worker(args: argparse.Namespace) -> optuna.study.Study:
    device = resolve_device(args.device)
    configure_runtime(args, device)
    loader_kwargs = build_loader_kwargs(args, device)
    study = create_study(args)
    spec = MODEL_SPECS[args.model]
    study.optimize(
        spec.objective_builder(args, device, loader_kwargs),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=args.show_progress,
    )
    return study


def main(
    default_model: str | None = None,
    *,
    include_model: bool = True,
    description: str = "Generic Fashion-MNIST HPO worker",
) -> None:
    args = parse_args(
        default_model=default_model,
        include_model=include_model,
        description=description,
    )
    study = run_worker(args)
    if args.show_progress or args.verbose_trainer:
        print_best_trial(study)


if __name__ == "__main__":
    main()