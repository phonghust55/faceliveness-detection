"""Streamlit demo (phương án B) - nhanh nhất để show GV.

Chạy: streamlit run webapp/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Cho phép import `src.*` khi chạy bằng streamlit từ project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from src.config import MODEL_PATH
from src.inference.predictor import LivenessPredictor


@st.cache_resource(show_spinner="Loading liveness model...")
def load_predictor() -> LivenessPredictor:
    return LivenessPredictor(model_path=MODEL_PATH)


class LivenessTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.predictor = load_predictor()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        result = self.predictor.predict(img)

        if result["found_face"]:
            x, y, w, h = result["bbox"]
            color = (0, 200, 0) if result["label"] == "real" else (0, 0, 220)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            text = f'{result["label"].upper()} {result["confidence"]*100:.0f}%'
            cv2.putText(
                img, text, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
        return img


st.set_page_config(page_title="Face Liveness Detection", layout="centered")
st.title("Face Liveness Detection")
st.caption("Demo môn Xác thực Sinh trắc học - MobileNetV2 + MediaPipe")

st.markdown(
    """
    **Hướng dẫn:**
    1. Bấm **START** ở dưới để cấp quyền webcam.
    2. Đưa khuôn mặt vào khung hình → khung xanh = REAL, khung đỏ = SPOOF.
    3. Thử in/đưa ảnh điện thoại → mô hình phải nhận ra là SPOOF.
    """
)

webrtc_streamer(
    key="liveness",
    video_transformer_factory=LivenessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
