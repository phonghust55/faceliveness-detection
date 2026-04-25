"""High-level wrapper dùng cho webapp / Streamlit.

Ý tưởng: load model 1 lần, expose method `predict(frame_bgr)` trả về:
    {
        "found_face": bool,
        "bbox": (x, y, w, h) | None,
        "label": "real" | "spoof",
        "confidence": float,
        "score_real": float,    # raw sigmoid score
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import DECISION_THRESHOLD, IMG_SIZE, MODEL_PATH
from ..models.loader import load_liveness_model
from ..preprocessing.face_detector import FaceDetector


class LivenessPredictor:
    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        threshold: float = DECISION_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.model = load_liveness_model(model_path)
        self.detector = FaceDetector()

    def _prep_face(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = face_rgb.astype("float32") / 255.0
        return np.expand_dims(face, axis=0)         # shape (1, H, W, 3)

    def predict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        face, bbox = self.detector.detect_and_crop(frame_bgr, target_size=IMG_SIZE)
        if face is None:
            return {
                "found_face": False,
                "bbox": None,
                "label": None,
                "confidence": 0.0,
                "score_real": 0.0,
            }
        x = self._prep_face(face)
        score_real = float(self.model.predict(x, verbose=0)[0, 0])
        is_real = score_real >= self.threshold
        # confidence là khoảng cách từ ngưỡng (chuẩn hoá về 0..1)
        confidence = float(score_real if is_real else 1.0 - score_real)
        return {
            "found_face": True,
            "bbox": list(bbox),
            "label": "real" if is_real else "spoof",
            "confidence": confidence,
            "score_real": score_real,
        }

    def close(self) -> None:
        self.detector.close()
