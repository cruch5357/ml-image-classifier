from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# CIFAR-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Repo root: .../ml-image-classifier
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess compatible con CIFAR-10 (32x32)
_preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def _build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    ResNet18 adaptada a CIFAR-10:
    - conv1: 3x3 stride1 padding1 (en vez de 7x7 stride2)
    - sin maxpool
    - fc a num_classes
    """
    model = models.resnet18(weights=None)  # no descarga pesos, solo arquitectura
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@lru_cache(maxsize=1)
def load_model() -> nn.Module:
    """
    Carga segura del modelo (checkpoint con state_dict) compatible con PyTorch 2.6+.
    """
    print("🔎 Loading model from:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first: python scripts/train.py"
        )

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Validaciones por si el archivo es otro formato
    if isinstance(ckpt, nn.Module):
        # Si por alguna razón aún guardaste el modelo completo, esto podría fallar
        # en PyTorch 2.6+ por seguridad. Dejamos mensaje claro:
        raise RuntimeError(
            "El archivo model.pt parece ser un modelo completo (pickle). "
            "Re-entrena y guarda como checkpoint con state_dict."
        )

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            "Formato de checkpoint inválido. Se esperaba dict con key 'state_dict'. "
            "Re-entrena y guarda como checkpoint."
        )

    num_classes = int(ckpt.get("num_classes", 10))
    model = _build_resnet18_cifar10(num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.inference_mode()
def predict_topk(img: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
    model = load_model()

    x = _preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)  # [1,3,32,32]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
    results: List[Tuple[str, float]] = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        results.append((CLASSES[i], float(p)))
    return results
