"""Train the deepfake detector on a real/fake image folder.

Example:
    python -m backend.ml.train --data data --epochs 5 --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import os

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_dataset
from .model import DeepfakeDetector


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, all_probs, all_targets = 0.0, [], []

    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_probs.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = sum(int(p == t) for p, t in zip(preds, all_targets)) / len(all_targets)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="root with train/ and val/ subfolders")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--out", default="checkpoints/detector.pt")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Backbone: {args.backbone}")

    train_ds = build_dataset(os.path.join(args.data, "train"), train=True)
    val_ds = build_dataset(os.path.join(args.data, "val"), train=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=2)

    model = DeepfakeDetector(args.backbone, pretrained=not args.no_pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader, device, criterion)
        print(f"[{epoch}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} auc {va_auc:.3f}")
        if va_auc >= best_auc:
            best_auc = va_auc
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone}, args.out)
            print(f"  saved {args.out} (val AUC {va_auc:.3f})")

    print(f"Done. Best val AUC: {best_auc:.3f}")


if __name__ == "__main__":
    main()
