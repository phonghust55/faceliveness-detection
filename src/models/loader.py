"""Helper load model với backward-compat cho checkpoint cũ.

Trong Keras 3:
- Model train trước commit fix-Lambda dùng `Lambda(preprocess_input)` → cần
  `safe_mode=False` + import `preprocess_input` để serializer tìm được.
- Model train sau commit fix-Lambda dùng `Rescaling` → load chuẩn không cần gì cả.

Hàm này thử lần lượt 2 cách để hoạt động với cả 2 loại checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

# QUAN TRỌNG: import để decorator @register_keras_serializable chạy trước khi load
from ..preprocessing.augmentation import preprocess_input  # noqa: F401


def load_liveness_model(weights_path: Path | str) -> tf.keras.Model:
    """Load model checkpoint, tự xử lý cả format cũ (Lambda) và mới (Rescaling)."""
    weights_path = str(weights_path)
    try:
        return tf.keras.models.load_model(weights_path, compile=False)
    except (TypeError, ValueError) as e:
        # Format cũ có Lambda(preprocess_input) → cần safe_mode=False
        print(f"[!] Load chuẩn thất bại ({type(e).__name__}). "
              f"Thử lại với safe_mode=False (cho checkpoint cũ).")
        return tf.keras.models.load_model(
            weights_path,
            compile=False,
            safe_mode=False,
            custom_objects={"preprocess_input": preprocess_input},
        )
