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

# id -> (path, class name, input shape, kwargs)
MODELS: Dict[str, Tuple[Path, str, Tuple[int, ...], Dict[str, Any]]] = {
    "densenet18": (ROOT / "modern_cnns" / "densenet.py", "DenseNet18", (1, 1, 96, 96), {}),
    "resnext50_32x4d": (ROOT / "modern_cnns" / "resnext.py", "ResNeXt50_32x4d", (1, 1, 96, 96), {}),
    "resnet_pre_act18": (ROOT / "modern_cnns" / "resnet_pre_act.py", "ResNet18", (1, 1, 96, 96), {}),
    "resnet18": (ROOT / "modern_cnns" / "resnet.py", "ResNet18", (1, 1, 96, 96), {}),
    "resnet50": (ROOT / "modern_cnns" / "resnet.py", "ResNet50", (1, 1, 96, 96), {}),
    "googlenet_bn": (ROOT / "modern_cnns" / "googlenet_batch_norm.py", "GoogleNetBN", (1, 1, 96, 96), {}),
    "googlenet": (ROOT / "modern_cnns" / "googlenet.py", "GoogLeNet", (1, 1, 96, 96), {}),
    "nin": (ROOT / "modern_cnns" / "nin.py", "NiN", (1, 1, 84, 84), {}),
    "vgg_improved": (ROOT / "modern_cnns" / "vgg_improved.py", "VGG", (1, 1, 84, 84), {"arch": VGG11_ARCH}),
    "alexnet": (ROOT / "modern_cnns" / "alexnet.py", "AlexNet", (1, 1, 224, 224), {}),
    "lenet": (ROOT / "lenet" / "lenet_modern.py", "LeNetModern", (1, 1, 28, 28), {}),
    "rnn_concise": (ROOT / "rnn" / "rnn_concise.py", "RNNConciseModel", (1, 48), {}),
    "lstm": (ROOT / "modern_rnns" / "lstm.py", "LSTMLM", (1, 48), {}),
    "gru": (ROOT / "modern_rnns" / "gru.py", "GRULM", (1, 48), {}),
    "vit": (
        ROOT / "attention_transformer" / "ViT.py",
        "ViT",
        (1, 1, 96, 96),
        {
            "img_size": 96,
            "patch_size": 16,
            "embed_dim": 512,
            "num_heads": 8,
            "mlp_hidden_dim": 2048,
            "num_blocks": 2,
            "num_classes": 10,
            "embed_dropout": 0.1,
            "blk_dropout": 0.1,
        },
    ),
}
