"""Shared model loading helpers for local profiling scripts."""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_class_from_path(path: Path, cls_name: str, *args: Any, **kwargs: Any) -> Any:
    spec = importlib.util.spec_from_file_location(cls_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, cls_name)
    return cls(*args, **kwargs)


def init_lazy_model(model: torch.nn.Module, h: int, w: int) -> None:
    """Run a dummy forward pass to initialize Lazy modules."""
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros((1, 1, h, w)))
