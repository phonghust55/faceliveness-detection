"""Mô hình Face Anti-Spoofing dùng Transfer Learning.

CHỌN MobileNetV2 vì:
1. Số tham số nhỏ (~3.5M) → train nhanh, suy luận nhanh trên CPU webapp.
2. Sử dụng Depthwise Separable Convolution → giảm FLOPs đáng kể so với ResNet50.
3. Pre-trained trên ImageNet → tận dụng feature low-level (cạnh, texture) rất phù hợp
   với bài toán anti-spoofing vốn dựa vào texture.
4. Cho phép chiến lược 2 giai đoạn: (1) train head, (2) fine-tune cả backbone.

Có thể đổi sang ResNet50 chỉ bằng cách thay 2 dòng `MobileNetV2` → `ResNet50`.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from ..config import DROPOUT_RATE, IMG_SIZE
from ..preprocessing.augmentation import build_train_augmentation


def build_model(
    img_size: int = IMG_SIZE,
    dropout: float = DROPOUT_RATE,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """Khởi tạo model: Input → Augment → Preprocess → MobileNetV2 → Head → Sigmoid.

    Args:
        img_size: kích thước ảnh đầu vào (square).
        dropout: tỷ lệ dropout trong head.
        use_augmentation: bật augmentation layer (tắt khi inference).

    Returns:
        Keras Model với output shape (batch, 1) - sigmoid score (0=spoof, 1=real).
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_face")

    # Pixel đầu vào ở range [0, 1]. Augmentation chỉ active ở training mode.
    x = inputs
    if use_augmentation:
        x = build_train_augmentation()(x)

    # Normalize về [-1, 1] theo chuẩn MobileNetV2.
    # Rescaling(scale=2, offset=-1)  ⇔  (x - 0.5) * 2  → tương đương preprocess_input cũ
    # Dùng Rescaling thay Lambda(custom_func) để serialize/deserialize sạch trong Keras 3.
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_preprocess")(x)

    # === Backbone ===
    backbone = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,         # Bỏ classifier ImageNet 1000 classes
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = False     # Stage 1: freeze toàn bộ backbone
    x = backbone(x, training=False)

    # === Head ===
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="fc_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="liveness_score")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LivenessNet_MobileNetV2")
    return model


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Tìm backbone (nested Functional model) trong outer model.

    Trong Keras 3, gán backbone._name sau khi tạo không thực sự đổi tên layer.
    Cách robust: backbone luôn là layer duy nhất là subclass của tf.keras.Model
    (nested model) hoặc layer có tên bắt đầu bằng 'mobilenet'/'resnet'.
    """
    candidates = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        or layer.name.startswith(("mobilenet", "resnet", "efficientnet"))
    ]
    if not candidates:
        raise RuntimeError(
            "Không tìm thấy backbone trong model. Layers có sẵn: "
            f"{[l.name for l in model.layers]}"
        )
    return candidates[0]


def unfreeze_backbone(model: tf.keras.Model, unfreeze_from_layer: int = 100) -> None:
    """Mở băng phần cuối của MobileNetV2 cho stage fine-tuning.

    MobileNetV2 có ~155 layer. Mở băng từ layer 100 trở đi tức là khoảng 1/3 cuối -
    nơi học các feature high-level (mặt, texture phức tạp) phù hợp tinh chỉnh cho
    domain-specific task của ta.
    """
    backbone = _find_backbone(model)
    print(f"[*] Unfreeze backbone '{backbone.name}' từ layer {unfreeze_from_layer} "
          f"(tổng {len(backbone.layers)} layers)")
    backbone.trainable = True
    for layer in backbone.layers[:unfreeze_from_layer]:
        layer.trainable = False
    # Luôn freeze BatchNorm để tránh phá vỡ statistics đã học từ ImageNet
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
