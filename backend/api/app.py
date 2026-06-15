"""FastAPI service for the deepfake detector.

Run:  uvicorn backend.api.app:app --reload --port 8000

Endpoints:
  GET  /health
  POST /predict/image   (multipart file)  -> verdict + base64 Grad-CAM overlay
  POST /predict/video   (multipart file)  -> verdict + per-frame timeline
"""
from __future__ import annotations

import base64
import io
import os
import tempfile

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ..ml.inference import ImagePredictor
from ..ml.model import DeepfakeDetector, load_checkpoint
from ..ml.video import analyze_video

CKPT = os.getenv("CKPT_PATH", "checkpoints/detector.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Deepfake & AI-Media Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup. Falls back to an untrained backbone so the API still
# boots for a UI demo — train.py produces the real checkpoint.
if os.path.exists(CKPT):
    _model = load_checkpoint(CKPT, DEVICE)
    _model_ready = True
else:
    _model = DeepfakeDetector(pretrained=False).to(DEVICE).eval()
    _model_ready = False

predictor = ImagePredictor(_model, DEVICE)


def _b64(image_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buf).decode() if ok else ""


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": _model_ready, "device": DEVICE}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded image.")
    result = predictor.predict(pil)
    overlay = predictor.explain(pil)
    result["heatmap"] = f"data:image/jpeg;base64,{_b64(overlay)}"
    result["model_trained"] = _model_ready
    return result


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = analyze_video(tmp_path, predictor)
        result["model_trained"] = _model_ready
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp_path)
