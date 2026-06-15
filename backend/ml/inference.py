"""Single-image inference with an explainability overlay."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import build_transforms
from .gradcam import GradCAM, overlay_heatmap
from .model import DeepfakeDetector


class ImagePredictor:
    def __init__(self, model: DeepfakeDetector, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.tf = build_transforms(train=False)
        self.cam = GradCAM(model)

    def _tensor(self, pil_img: Image.Image) -> torch.Tensor:
        return self.tf(pil_img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> dict:
        prob_fake = self.model.predict_proba(self._tensor(pil_img)).item()
        return {
            "label": "fake" if prob_fake >= 0.5 else "real",
            "fake_probability": round(prob_fake, 4),
            "confidence": round(max(prob_fake, 1 - prob_fake), 4),
        }

    def explain(self, pil_img: Image.Image) -> np.ndarray:
        """Return a BGR image with the Grad-CAM heatmap overlaid."""
        x = self._tensor(pil_img)
        cam = self.cam(x)  # gradients needed, so no no_grad here
        original = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return overlay_heatmap(original, cam)
