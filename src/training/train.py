"""Training pipeline 2 giai đoạn (two-stage transfer learning).

Stage 1: Freeze backbone, chỉ train head với lr lớn (1e-3).
Stage 2: Unfreeze ~1/3 cuối backbone, fine-tune với lr rất nhỏ (1e-5)
         để không phá vỡ feature ImageNet đã học.

Đây là pattern chuẩn được khuyến nghị bởi tài liệu Keras Transfer Learning.
"""
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from ..config import (
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_FINETUNE,
    EPOCHS_HEAD,
    LEARNING_RATE_FINETUNE,
    LEARNING_RATE_HEAD,
    REPORTS_DIR,
)
from ..models.liveness_model import build_model, unfreeze_backbone
from .data_loader import compute_class_weight, get_datasets


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]


def _make_callbacks(stage: str) -> list[tf.keras.callbacks.Callback]:
    ckpt_path = CHECKPOINT_DIR / f"best_model_{stage}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _merge_history(h1: dict, h2: dict) -> dict:
    """Nối history 2 stage để vẽ biểu đồ liền mạch."""
    merged = {k: list(v) for k, v in h1.items()}
    for k, v in h2.items():
        merged.setdefault(k, []).extend(list(v))
    return merged


def train(skip_stage1: bool = False) -> tuple[tf.keras.Model, dict]:
    """Train 2 giai đoạn, trả về (model, full_history_dict).

    Args:
        skip_stage1: nếu True, load weights từ checkpoints/best_model_head.keras
                     và bỏ qua Stage 1 (dùng khi resume sau lỗi ở Stage 2).
    """
    train_ds, val_ds, _ = get_datasets()

    # Class weight để bù trừ imbalance (LCC-FASD: real:spoof ≈ 1:5.8 trong train)
    class_weight = compute_class_weight()

    model = build_model(use_augmentation=True)
    print(model.summary())

    history_1_dict: dict = {}
    head_ckpt = CHECKPOINT_DIR / "best_model_head.keras"

    # ---- STAGE 1: Train head ----
    if skip_stage1 and head_ckpt.exists():
        print("\n" + "=" * 60)
        print(f"BỎ QUA STAGE 1, load weights từ: {head_ckpt}")
        print("=" * 60)
        model.load_weights(str(head_ckpt))
    else:
        if skip_stage1:
            print(f"[!] Yêu cầu skip_stage1 nhưng không thấy {head_ckpt} → "
                  f"vẫn chạy Stage 1.")
        print("\n" + "=" * 60)
        print("STAGE 1: Train HEAD (backbone frozen)")
        print("=" * 60)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=METRICS,
        )
        history_1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_HEAD,
            class_weight=class_weight,
            callbacks=_make_callbacks("head"),
            verbose=1,
        )
        history_1_dict = history_1.history

    # ---- STAGE 2: Fine-tune ----
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tune backbone (unfreeze last layers)")
    print("=" * 60)
    unfreeze_backbone(model, unfreeze_from_layer=100)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weight,
        callbacks=_make_callbacks("finetune"),
        verbose=1,
    )

    # Lưu model cuối cùng
    final_path = CHECKPOINT_DIR / "best_model.keras"
    model.save(final_path)
    print(f"\n[+] Đã lưu model cuối tại: {final_path}")

    # Gộp & lưu history (dùng cho visualize.py)
    full_history = _merge_history(history_1_dict, history_2.history)
    history_path = REPORTS_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"[+] Đã lưu history tại: {history_path}")

    return model, full_history
