"""Central registry of local models for profiling utilities."""
from pathlib import Path
from typing import Any, Dict, Tuple

from tools.model_loader import ROOT

# Arch definitions reused across scripts
VGG11_ARCH = [
    (1, 64, False),
    (1, 128, False),
    (2, 256, True),
    (2, 512, True),
    (2, 512, True),
]

# id -> (path, class name, H, W, kwargs)
MODELS: Dict[str, Tuple[Path, str, int, int, Dict[str, Any]]] = {
    "googlenet": (ROOT / "modern_cnns" / "googlenet.py", "GoogLeNet", 96, 96, {}),
    "nin": (ROOT / "modern_cnns" / "nin.py", "NiN", 84, 84, {}),
    "vgg_improved": (ROOT / "modern_cnns" / "vgg_improved.py", "VGG", 84, 84, {"arch": VGG11_ARCH}),
    "alexnet": (ROOT / "modern_cnns" / "alexnet.py", "AlexNet", 224, 224, {}),
    "lenet": (ROOT / "lenet" / "lenet_modern.py", "LeNetModern", 28, 28, {}),
}
