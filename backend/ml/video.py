"""Video deepfake scoring.

Strategy (standard for face-swap deepfakes):
  1. Sample N evenly-spaced frames.
  2. Detect the largest face in each frame (OpenCV Haar cascade — ships with
     opencv-python, no extra download). Fall back to the whole frame if none.
  3. Classify each crop with the image model.
  4. Aggregate per-frame fake-probabilities into one verdict.

Deepfakes are rarely uniform across a clip, so we report the mean *and* the
peak frame, and flag the video as fake if a meaningful fraction of frames
look manipulated.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .inference import ImagePredictor

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _largest_face(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame_bgr
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))  # a little context around the face
    y0, y1 = max(0, y - pad), min(frame_bgr.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(frame_bgr.shape[1], x + w + pad)
    return frame_bgr[y0:y1, x0:x1]


def analyze_video(path: str, predictor: ImagePredictor, num_frames: int = 24,
                  fake_frame_threshold: float = 0.5,
                  fake_video_ratio: float = 0.35) -> dict:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise ValueError("Could not read the video or it has no frames.")

    indices = np.linspace(0, total - 1, min(num_frames, total)).astype(int)
    frame_scores = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        face = _largest_face(frame)
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        prob = predictor.model.predict_proba(predictor._tensor(pil)).item()
        frame_scores.append({"frame": int(idx), "fake_probability": round(prob, 4)})
    cap.release()

    if not frame_scores:
        raise ValueError("No frames could be decoded from the video.")

    probs = [f["fake_probability"] for f in frame_scores]
    fake_frames = sum(p >= fake_frame_threshold for p in probs)
    ratio = fake_frames / len(probs)
    peak = max(frame_scores, key=lambda f: f["fake_probability"])

    return {
        "label": "fake" if ratio >= fake_video_ratio else "real",
        "mean_fake_probability": round(float(np.mean(probs)), 4),
        "fake_frame_ratio": round(ratio, 4),
        "frames_analyzed": len(probs),
        "peak_frame": peak,
        "timeline": frame_scores,
    }
