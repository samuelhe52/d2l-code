"""Print architecture summaries for local models using torchinfo.

Run:
    python tools/model_summary.py --models googlenet,nin

Requires:
    pip install torchinfo
"""
from pathlib import Path
import sys

import torch
from torchinfo import summary

sys.path.append(str(Path(__file__).resolve().parent))
from model_loader import ROOT, init_lazy_model, load_class_from_path
from model_registry import MODELS
import argparse


def summarize(model: torch.nn.Module, input_shape: tuple[int, ...]) -> str:
    init_lazy_model(model, input_shape)
    return summary(
        model,
        input_size=input_shape,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        verbose=0,
    ).__str__()


def format_input_shape(input_shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in input_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model summary printer")
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
        shape_str = format_input_shape(input_shape)
        print(f"===== {mid} (input={shape_str}) =====")
        print(summarize(model, input_shape))
        print()


if __name__ == "__main__":
    main()
