"""Face detector dùng MediaPipe **Tasks API** (mediapipe>=0.10.22).

Tại sao MediaPipe?
- Tốc độ ~30 FPS trên CPU laptop, không cần GPU.
- Trả về bbox + score, độ chính xác cao hơn Haar Cascade.
- So sánh với MTCNN: MediaPipe nhẹ hơn nhiều (model ~225 KB vs MTCNN ~6 MB).

Lưu ý kỹ thuật:
- MediaPipe phiên bản >= 0.10.22 đã loại bỏ legacy `mp.solutions` API.
- Module này dùng `mp.tasks.vision.FaceDetector` — model `blaze_face_short_range.tflite`
  được tự động download lần đầu vào thư mục `checkpoints/`.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..config import CHECKPOINT_DIR, FACE_MARGIN, IMG_SIZE, MIN_DETECTION_CONFIDENCE

# BlazeFace short-range (tối ưu cho khoảng cách <2m, đúng usecase webcam)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_MODEL_PATH = CHECKPOINT_DIR / "blaze_face_short_range.tflite"


def _ensure_model() -> Path:
    """Download model lần đầu nếu chưa có (~225 KB)."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Tải MediaPipe face detector model về {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[+] Đã tải xong.")
    return _MODEL_PATH


class FaceDetector:
    """Wrapper quanh MediaPipe Face Detector (Tasks API).

    Sử dụng:
        detector = FaceDetector()
        face_crop, bbox = detector.detect_and_crop(image_bgr)
    """

    def __init__(self, min_confidence: float = MIN_DETECTION_CONFIDENCE) -> None:
        model_path = _ensure_model()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Trả về bbox (x, y, w, h) của khuôn mặt CONFIDENCE cao nhất, hoặc None."""
        if image_bgr is None or image_bgr.size == 0:
            return None

        # MediaPipe Tasks API yêu cầu mp.Image với format SRGB (= RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self._detector.detect(mp_image)
        if not result.detections:
            return None

        # Lấy detection có score cao nhất
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bb = best.bounding_box  # đã ở pixel (origin_x, origin_y, width, height)
        x, y = int(bb.origin_x), int(bb.origin_y)
        bw, bh = int(bb.width), int(bb.height)

        # Mở rộng bbox với margin (giữ context: trán, cằm, tóc)
        h, w = image_bgr.shape[:2]
        mx = int(bw * FACE_MARGIN)
        my = int(bh * FACE_MARGIN)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)
        return x1, y1, x2 - x1, y2 - y1

    def detect_and_crop(
        self, image_bgr: np.ndarray, target_size: int = IMG_SIZE
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Detect face → crop → resize.

        Returns:
            (face_bgr_resized, bbox) hoặc (None, None) nếu không tìm thấy face.
        """
        bbox = self.detect(image_bgr)
        if bbox is None:
            return None, None
        x, y, w, h = bbox
        face = image_bgr[y : y + h, x : x + w]
        if face.size == 0:
            return None, None
        face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return face, bbox

    def close(self) -> None:
        self._detector.close()
