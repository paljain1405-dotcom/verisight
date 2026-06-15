"""Grad-CAM heatmaps so the detector can show *where* it sees manipulation.

This is the explainability piece — instead of a bare "87% fake", the UI
overlays a heatmap highlighting the regions that pushed the score up.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import FAKE, find_last_conv


class GradCAM:
    """Minimal Grad-CAM for a single target class."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model.eval()
        self.target_layer = target_layer or find_last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int = FAKE) -> np.ndarray:
        """Return a [H, W] heatmap in [0, 1] for a single-image batch (B=1)."""
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Channel weights = global-average-pooled gradients.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over the original image (both BGR, same size)."""
    cam_resized = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, image_bgr, 1 - alpha, 0)
