"""Data augmentation pipeline.

Triết lý augmentation cho Face Anti-Spoofing:
- Bài toán phụ thuộc vào TEXTURE (lỗ chân lông, moire pattern của màn hình, v.v.)
- KHÔNG nên augment quá mạnh làm mất texture: tránh blur mạnh, tránh elastic transform.
- Lật ngang (horizontal flip) OK vì khuôn mặt khá đối xứng.
- Brightness/Contrast: BẮT BUỘC vì webcam người dùng có ánh sáng đa dạng.
- Rotation nhỏ (±15°): mô phỏng nghiêng đầu.
- KHÔNG flip dọc (vertical flip) vì không tự nhiên.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from tensorflow.keras import layers


def build_train_augmentation() -> tf.keras.Sequential:
    """Augmentation cho TẬP TRAIN. Áp dụng on-the-fly trên GPU/CPU."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.04),          # ±15°  (15/360 ≈ 0.04)
            layers.RandomZoom(height_factor=0.1),
            layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),
            layers.RandomContrast(factor=0.2),
        ],
        name="train_augmentation",
    )


@keras.saving.register_keras_serializable(package="liveness", name="preprocess_input")
def preprocess_input(x: tf.Tensor) -> tf.Tensor:
    """Chuẩn hóa pixel theo MobileNetV2 (-1, 1).

    GIỮ LẠI để load các checkpoint cũ (best_model.keras đã train trước commit này).
    Code mới dùng layers.Rescaling(scale=2.0, offset=-1.0) thay thế (sạch hơn,
    không cần decorator để serialize).

    Note: tf.keras.applications.mobilenet_v2.preprocess_input nhận pixel [0, 255].
    Pipeline tf.data của ta đã decode về [0, 1] nên cần nhân lại 255 trước khi gọi.
    Để đơn giản, ta tự normalize về [-1, 1].
    """
    return (x - 0.5) * 2.0
