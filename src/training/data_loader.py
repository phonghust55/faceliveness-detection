"""tf.data pipeline đọc ảnh từ thư mục đã preprocess.

Sử dụng `image_dataset_from_directory` của Keras vì:
- Tự động đọc nhãn từ tên thư mục con (real/, spoof/).
- Tích hợp sẵn với `tf.data` → prefetch, batch, shuffle hiệu năng cao.
- Rất ít code, dễ trình bày trong báo cáo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from ..config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, PROCESSED_DIR, RANDOM_SEED


def _make_ds(split_dir: Path, shuffle: bool) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(split_dir),
        labels="inferred",
        label_mode="binary",                 # output 0/1, phù hợp sigmoid + BCE
        class_names=CLASS_NAMES,             # bảo đảm thứ tự: spoof=0, real=1
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=shuffle,
        seed=RANDOM_SEED,
        interpolation="bilinear",
    )
    # Chuẩn hóa pixel về [0, 1]; đoạn map nhẹ vì mọi xử lý còn lại nằm trong model
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def get_datasets(processed_dir: Path = PROCESSED_DIR) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Trả về (train_ds, val_ds, test_ds)."""
    train_ds = _make_ds(processed_dir / "train", shuffle=True)
    val_ds = _make_ds(processed_dir / "val", shuffle=False)
    test_ds = _make_ds(processed_dir / "test", shuffle=False)
    return train_ds, val_ds, test_ds


def compute_class_weight(processed_dir: Path = PROCESSED_DIR) -> Dict[int, float]:
    """Tính class_weight kiểu sklearn 'balanced' từ phân bố train set.

    Công thức:  w_c = N_total / (n_classes * N_c)
    → Lớp ít mẫu hơn nhận trọng số lớn hơn, giúp loss công bằng giữa real/spoof.
    Bắt buộc khi LCC-FASD có tỉ lệ real:spoof ≈ 1:5.8 trong train set.

    Returns:
        dict {0: w_spoof, 1: w_real}  (khớp với mapping CLASS_NAMES)
    """
    train_dir = processed_dir / "train"
    counts = {}
    for idx, cls in enumerate(CLASS_NAMES):
        n = len(list((train_dir / cls).glob("*.jpg")))
        counts[idx] = n
    total = sum(counts.values())
    n_cls = len(counts)
    weights = {i: total / (n_cls * c) if c > 0 else 1.0 for i, c in counts.items()}
    print(f"[*] Phân bố train set: {dict(zip(CLASS_NAMES, counts.values()))}")
    print(f"[*] class_weight     : {dict(zip(CLASS_NAMES, weights.values()))}")
    return weights
