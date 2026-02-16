from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # CIFAR-10 normalization
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = ROOT / "data"
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # ✅ Opcional (recomendado en CPU): entrenar con subset para que sea rápido y demo-friendly
    # Comenta estas 2 líneas si quieres usar los 50k completos
    train_ds = Subset(train_ds, range(8000))   # 8k en vez de 50k
    test_ds = Subset(test_ds, range(2000))     # 2k para evaluación más rápida

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    print("✅ CIFAR-10 listo:", len(train_ds), "train |", len(test_ds), "test")
    print("✅ Device:", DEVICE)

    # Transfer learning: ResNet18 adaptada a 32x32
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    # ✅ Congelar TODO y entrenar solo la última capa (mucho más rápido en CPU)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_acc = 0.0
    epochs = 2  # ✅ rápido para demo; sube a 5-10 si tienes GPU o paciencia
    log_every = 100

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ✅ logs para ver progreso (evita sensación de “freeze”)
            if step % log_every == 0 or step == 1:
                print(f"  epoch {epoch}/{epochs} | step {step}/{len(train_loader)} | loss={running_loss/step:.4f}")

        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch} done | avg_loss={running_loss/len(train_loader):.4f} | acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "num_classes": 10,
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, MODELS_DIR / "model.pt")

            model = model.to(DEVICE)
            print("✅ Saved new best model -> models/model.pt")

    print(f"✅ Best acc: {best_acc:.4f}")
    print(f"✅ Saved to: {MODELS_DIR / 'model.pt'}")


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


if __name__ == "__main__":
    main()
