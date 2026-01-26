"""Quick FLOPs/params profiling for local models using ptflops.

Run:
    python tools/profile_flops.py --models googlenet,nin

Requires:
    pip install ptflops
"""
from pathlib import Path
from typing import Any, Dict, Tuple
import sys
import argparse

import torch
from ptflops import get_model_complexity_info

sys.path.append(str(Path(__file__).resolve().parent))
from model_loader import ROOT, init_lazy_model, load_class_from_path
from model_registry import MODELS


def measure(model: torch.nn.Module, h: int, w: int) -> Tuple[str, str]:
    model.eval()
    init_lazy_model(model, h, w)
    macs, params = get_model_complexity_info(
        model,
        (1, h, w),  # C, H, W
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params


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
        path, cls_name, h, w, kwargs = MODELS[mid]
        model = load_class_from_path(path, cls_name, **kwargs)
        macs, params = measure(model, h, w)
        print(f"{mid}: MACs {macs}, Params {params}, input=1x{h}x{w}")
    print("\nNote: MACs are multiply-accumulates. Multiply by ~2 for FLOPs (mul+add).")


if __name__ == "__main__":
    main()
