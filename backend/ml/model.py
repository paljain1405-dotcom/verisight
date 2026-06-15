"""Deepfake / AI-generated media detector.

A thin wrapper around a timm CNN backbone with a 2-class head
(0 = real, 1 = fake). Kept deliberately simple so the same
checkpoint works for both still images and per-frame video scoring.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install dependencies first: pip install -r requirements.txt") from exc

# Label convention used everywhere in the project.
CLASSES = ["real", "fake"]
REAL, FAKE = 0, 1


class DeepfakeDetector(nn.Module):
    """CNN classifier that outputs a real/fake logit pair."""

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fake-probability for each item in the batch (shape [B])."""
        self.eval()
        probs = torch.softmax(self.forward(x), dim=1)
        return probs[:, FAKE]


def find_last_conv(model: nn.Module) -> nn.Module:
    """Return the last 2D conv layer — the target layer for Grad-CAM.

    Works across timm backbones (EfficientNet, ResNet, Xception, etc.)
    without hard-coding a layer name.
    """
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found; Grad-CAM needs a convolutional backbone.")
    return last


def load_checkpoint(path: str, device: str = "cpu") -> DeepfakeDetector:
    """Rebuild a model from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device)
    model = DeepfakeDetector(backbone=ckpt.get("backbone", "efficientnet_b0"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
