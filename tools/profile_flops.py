"""Quick FLOPs/params profiling for local models using ptflops.

Run:
    python tools/profile_flops.py --models googlenet,nin

Requires:
    pip install ptflops
"""
from pathlib import Path
import sys
import argparse

import torch
from ptflops import get_model_complexity_info

sys.path.append(str(Path(__file__).resolve().parent))
from model_loader import ROOT, init_lazy_model, load_class_from_path
from model_registry import MODELS


def measure(model: torch.nn.Module, input_shape: tuple[int, ...]) -> tuple[str, str]:
    model.eval()
    init_lazy_model(model, input_shape)
    input_res = input_shape[1:] if len(input_shape) > 1 else input_shape
    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params


def format_input_shape(input_shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in input_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="FLOPs/params profiler")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model ids. Defaults to all.",
    )
    args = parser.parse_args()

    selected_ids = list(MODELS.keys()) if not args.models else [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected_ids if m not in MODELS]
    if unknown:
        known = ", ".join(MODELS.keys())
        raise SystemExit(f"Unknown model ids: {unknown}. Known: {known}")

    for mid in selected_ids:
        path, cls_name, input_shape, kwargs = MODELS[mid]
        model = load_class_from_path(path, cls_name, **kwargs)
        macs, params = measure(model, input_shape)
        shape_str = format_input_shape(input_shape)
        print(f"{mid}: MACs {macs}, Params {params}, input={shape_str}")
    print("\nNote: MACs are multiply-accumulates. Multiply by ~2 for FLOPs (mul+add).")


if __name__ == "__main__":
    main()
