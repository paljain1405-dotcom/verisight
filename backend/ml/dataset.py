"""Data loading for real-vs-fake image classification.

Expected layout (works with CIFAKE, the 140k Real-vs-Fake set, or your own):

    data/
      train/
        real/  *.jpg
        fake/  *.jpg
      val/
        real/  *.jpg
        fake/  *.jpg
"""
from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
# ImageNet stats — backbones are pretrained on it.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            # JPEG-style compression is a strong real/fake cue — augment lightly.
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_dataset(root: str, train: bool) -> ImageFolder:
    """ImageFolder that enforces the real=0, fake=1 ordering."""
    ds = ImageFolder(root, transform=build_transforms(train))
    # ImageFolder sorts class names alphabetically => fake=0, real=1.
    # Remap so it matches the project's REAL=0 / FAKE=1 convention.
    if ds.class_to_idx.get("real") != 0:
        remap = {old: (0 if cls == "real" else 1) for cls, old in ds.class_to_idx.items()}
        ds.samples = [(p, remap[t]) for p, t in ds.samples]
        ds.targets = [remap[t] for t in ds.targets]
        ds.class_to_idx = {"real": 0, "fake": 1}
        ds.classes = ["real", "fake"]
    return ds
