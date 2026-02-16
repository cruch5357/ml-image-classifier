from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

# CIFAR-10 classes
CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "model.pt"

# Preprocess compatible con CIFAR-10 (32x32)
_preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

_model = None

def load_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first: python scripts/train.py"
        )

    device = torch.device("cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    _model = model
    return _model

@torch.inference_mode()
def predict_topk(img: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
    model = load_model()
    x = _preprocess(img.convert("RGB")).unsqueeze(0)  # [1,3,32,32]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
    results = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        results.append((CLASSES[i], float(p)))
    return results
