"""Pydantic schemas cho FastAPI."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    found_face: bool = Field(..., description="True nếu detect được khuôn mặt")
    bbox: Optional[list[int]] = Field(
        None, description="Bounding box [x, y, w, h] toạ độ pixel trên ảnh gốc"
    )
    label: Optional[str] = Field(None, description='"real" hoặc "spoof"')
    confidence: float = Field(..., ge=0.0, le=1.0)
    score_real: float = Field(..., ge=0.0, le=1.0, description="Sigmoid score raw")
    inference_ms: float = Field(..., description="Thời gian xử lý (ms)")
