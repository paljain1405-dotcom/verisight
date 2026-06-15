"""Reshape a flat real/fake image collection into train/ and val/ splits.

Many AI-generated-image datasets (CIFAKE, the 140k Real-vs-Fake set) ship as
two folders of images. This script copies them into the layout train.py expects:

    data/train/{real,fake}/   data/val/{real,fake}/

Usage:
    python scripts/prepare_data.py --real path/to/real --fake path/to/fake --out data
"""
from __future__ import annotations

import argparse
import os
import random
import shutil


def split_class(src_dir: str, out_root: str, cls: str, val_ratio: float, seed: int):
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    for split, subset in (("val", files[:n_val]), ("train", files[n_val:])):
        dest = os.path.join(out_root, split, cls)
        os.makedirs(dest, exist_ok=True)
        for f in subset:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dest, f))
    print(f"  {cls}: {len(files) - n_val} train / {n_val} val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="folder of authentic images")
    ap.add_argument("--fake", required=True, help="folder of AI-generated / manipulated images")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Building splits in {args.out}/ (val ratio {args.val_ratio})")
    split_class(args.real, args.out, "real", args.val_ratio, args.seed)
    split_class(args.fake, args.out, "fake", args.val_ratio, args.seed)
    print("Done. Train with:  python -m backend.ml.train --data", args.out)


if __name__ == "__main__":
    main()
