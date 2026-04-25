"""Preprocessing pipeline: đọc ảnh raw → crop face → save thành processed/.

Hỗ trợ 2 chế độ:
1. **LCC-FASD official splits** (khuyến nghị):
   data/LCC_FASD_training    → data/processed/train
   data/LCC_FASD_development  → data/processed/val
   data/LCC_FASD_evaluation   → data/processed/test
   → Giữ nguyên split của tác giả → kết quả so sánh được với benchmark trong paper.

2. **Generic mode** (fallback): đọc data/raw/{real,spoof}/ rồi chia 70/15/15 theo seed.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from ..config import (
    CLASS_NAMES,
    DATA_DIR,
    IMG_SIZE,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    SPLIT_RATIO,
)
from .face_detector import FaceDetector

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Mapping cho LCC-FASD: {tên thư mục con của data/} → {tên split output}
LCC_FASD_SPLIT_MAP = {
    "LCC_FASD_training": "train",
    "LCC_FASD_development": "val",
    "LCC_FASD_evaluation": "test",
}


# ----------------------- helpers -----------------------
def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]


def _crop_and_save(
    files: List[Path],
    dst_dir: Path,
    detector: FaceDetector,
    desc: str,
) -> tuple[int, int]:
    """Crop face cho 1 list ảnh, lưu vào dst_dir. Trả về (saved, skipped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    for idx, fp in enumerate(tqdm(files, desc=desc, unit="img")):
        img = cv2.imread(str(fp))
        if img is None:
            skipped += 1
            continue
        face, _ = detector.detect_and_crop(img, target_size=IMG_SIZE)
        if face is None:
            skipped += 1
            continue
        cv2.imwrite(str(dst_dir / f"{idx:06d}.jpg"), face)
        saved += 1
    return saved, skipped


def _print_stats(all_stats: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("THỐNG KÊ PREPROCESSING")
    print("=" * 70)
    print(f"{'split':<8} {'class':<6} {'raw':>7} {'saved':>7} {'skipped':>9}")
    print("-" * 70)
    for s in all_stats:
        print(
            f"{s['split']:<8} {s['class']:<6} {s['total_raw']:>7} "
            f"{s['saved']:>7} {s['skipped']:>9}"
        )
    print("=" * 70)


# ----------------------- mode 1: LCC-FASD official splits -----------------------
def detect_lcc_fasd(data_dir: Path) -> bool:
    """Trả về True nếu phát hiện cấu trúc LCC-FASD trong data_dir."""
    return all((data_dir / name).is_dir() for name in LCC_FASD_SPLIT_MAP)


def run_from_lcc_fasd(
    data_dir: Path = DATA_DIR,
    out_dir: Path = PROCESSED_DIR,
    clean: bool = True,
) -> None:
    """Preprocess dùng split chính thức của LCC-FASD."""
    if clean and out_dir.exists():
        print(f"[*] Xóa thư mục cũ: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    all_stats: list[dict] = []
    try:
        for src_name, split_name in LCC_FASD_SPLIT_MAP.items():
            for cls in CLASS_NAMES:
                src_dir = data_dir / src_name / cls
                files = _list_images(src_dir)
                if not files:
                    print(f"[!] CẢNH BÁO: không tìm thấy ảnh trong {src_dir}")
                    continue
                dst_dir = out_dir / split_name / cls
                saved, skipped = _crop_and_save(
                    files, dst_dir, detector,
                    desc=f"[{split_name}/{cls}]",
                )
                all_stats.append({
                    "split": split_name, "class": cls,
                    "total_raw": len(files), "saved": saved, "skipped": skipped,
                })
    finally:
        detector.close()

    _print_stats(all_stats)
    _warn_imbalance(all_stats)


# ----------------------- mode 2: generic raw/{real,spoof}/ -----------------------
def _split_files(files: List[Path]) -> dict[str, List[Path]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def run_from_raw(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    clean: bool = True,
) -> None:
    """Preprocess từ data/raw/{real,spoof}/ - tự chia train/val/test."""
    if clean and out_dir.exists():
        print(f"[*] Xóa thư mục cũ: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    all_stats: list[dict] = []
    try:
        for cls in CLASS_NAMES:
            files = _list_images(raw_dir / cls)
            if not files:
                raise FileNotFoundError(
                    f"Không tìm thấy ảnh trong {raw_dir / cls}. "
                    f"Hãy đặt dataset đúng cấu trúc: {raw_dir}/real/, {raw_dir}/spoof/"
                )
            splits = _split_files(files)
            for split_name, split_files in splits.items():
                dst_dir = out_dir / split_name / cls
                saved, skipped = _crop_and_save(
                    split_files, dst_dir, detector,
                    desc=f"[{split_name}/{cls}]",
                )
                all_stats.append({
                    "split": split_name, "class": cls,
                    "total_raw": len(split_files), "saved": saved, "skipped": skipped,
                })
    finally:
        detector.close()

    _print_stats(all_stats)
    _warn_imbalance(all_stats)


# ----------------------- entry point -----------------------
def run(out_dir: Path = PROCESSED_DIR, clean: bool = True) -> None:
    """Auto-detect cấu trúc dataset và gọi đúng pipeline."""
    if detect_lcc_fasd(DATA_DIR):
        print("[*] Phát hiện cấu trúc LCC-FASD → dùng split chính thức của tác giả")
        run_from_lcc_fasd(data_dir=DATA_DIR, out_dir=out_dir, clean=clean)
    elif (RAW_DIR / "real").is_dir() and (RAW_DIR / "spoof").is_dir():
        print(f"[*] Phát hiện cấu trúc generic tại {RAW_DIR} → tự chia 70/15/15")
        run_from_raw(raw_dir=RAW_DIR, out_dir=out_dir, clean=clean)
    else:
        raise FileNotFoundError(
            "Không tìm thấy cấu trúc dataset hợp lệ. Cần một trong hai:\n"
            f"  (A) {DATA_DIR}/LCC_FASD_{{training,development,evaluation}}/{{real,spoof}}/\n"
            f"  (B) {RAW_DIR}/{{real,spoof}}/"
        )


# ----------------------- imbalance warning -----------------------
def _warn_imbalance(stats: list[dict]) -> None:
    """In cảnh báo nếu split nào đó imbalance nặng (giúp người dùng nhận biết)."""
    by_split: dict[str, dict[str, int]] = {}
    for s in stats:
        by_split.setdefault(s["split"], {})[s["class"]] = s["saved"]

    msgs = []
    for split, counts in by_split.items():
        n_real = counts.get("real", 0)
        n_spoof = counts.get("spoof", 0)
        if n_real == 0 or n_spoof == 0:
            continue
        ratio = max(n_real, n_spoof) / min(n_real, n_spoof)
        if ratio >= 3:
            major = "spoof" if n_spoof > n_real else "real"
            msgs.append(f"  - {split}: {major} nhiều gấp {ratio:.1f}× lớp còn lại")
    if msgs:
        print("\n[!] CẢNH BÁO: dataset MẤT CÂN BẰNG:")
        for m in msgs:
            print(m)
        print("    → Training sẽ tự dùng class_weight để bù trừ.")
        print("    → Khi đánh giá, hãy ưu tiên FAR/FRR/EER/HTER thay vì Accuracy.\n")
