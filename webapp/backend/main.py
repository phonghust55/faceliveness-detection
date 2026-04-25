"""FastAPI backend cho real-time Face Liveness Detection.

Architecture:
    [Browser]                          [FastAPI Backend]
    ┌──────────────────┐              ┌──────────────────────────┐
    │ getUserMedia     │              │ POST /predict            │
    │  → <video>       │   JPEG bytes │   1. decode img          │
    │  → <canvas>      │ ───────────► │   2. FaceDetector(MP)    │
    │  → toBlob('jpeg')│   ~10 KB     │   3. crop+resize 224     │
    │     mỗi 200ms    │              │   4. MobileNetV2.predict │
    │                  │ ◄─────────── │   5. return JSON         │
    │ Vẽ bbox + label  │   JSON       │                          │
    └──────────────────┘              └──────────────────────────┘

Tại sao gửi từng frame qua HTTP thay vì WebSocket?
- Đơn giản, dễ debug.
- Anti-spoofing chỉ cần ~5 FPS là đủ (không như object tracking).
- Nếu cần FPS cao hơn → có thể nâng cấp lên WebSocket sau.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import API_HOST, API_PORT, MODEL_PATH
from src.inference.predictor import LivenessPredictor

from .schemas import PredictionResponse


# ---------- Lifespan: load model 1 lần khi server start ----------
predictor: LivenessPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Không tìm thấy model tại {MODEL_PATH}. Hãy train trước: "
            f"python scripts/02_train.py"
        )
    print(f"[*] Loading model from {MODEL_PATH} ...")
    predictor = LivenessPredictor(model_path=MODEL_PATH)
    print("[+] Model loaded. API ready.")
    yield
    if predictor:
        predictor.close()


app = FastAPI(
    title="Face Liveness Detection API",
    description="API phát hiện khuôn mặt thật/giả phục vụ môn Biometrics",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS để frontend (mở từ file://) gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>Face Liveness Detection API đang chạy</h2>
    <p>POST một file ảnh tới <code>/predict</code> để nhận kết quả.</p>
    <p>Mở <code>webapp/frontend/index.html</code> để dùng webcam demo.</p>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Nhận 1 frame ảnh (JPEG/PNG), trả về kết quả liveness."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Decode bytes → BGR ndarray
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Không decode được ảnh")

    t0 = time.perf_counter()
    result = predictor.predict(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        found_face=result["found_face"],
        bbox=result["bbox"],
        label=result["label"],
        confidence=result["confidence"],
        score_real=result["score_real"],
        inference_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host=API_HOST, port=API_PORT, reload=True)
