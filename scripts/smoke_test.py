"""Smoke test nhanh: kiểm tra dataset detection + face crop trên vài ảnh.

Mục đích: verify pipeline trước khi chạy preprocess full (mất ~10-30 phút).
Chạy: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import DATA_DIR
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.prepare_dataset import LCC_FASD_SPLIT_MAP, detect_lcc_fasd

N_SAMPLES = 5


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # 1. Auto-detect dataset
    if not detect_lcc_fasd(DATA_DIR):
        print("[X] Không phát hiện cấu trúc LCC-FASD trong", DATA_DIR)
        sys.exit(1)
    print(f"[OK] Đã phát hiện cấu trúc LCC-FASD trong {DATA_DIR}")

    # 2. Test face detector trên vài ảnh từ mỗi split/class
    detector = FaceDetector()
    total_ok = 0
    total_fail = 0
    try:
        for src_name, split_name in LCC_FASD_SPLIT_MAP.items():
            for cls in ("real", "spoof"):
                folder = DATA_DIR / src_name / cls
                files = sorted(folder.glob("*"))[:N_SAMPLES]
                ok = 0
                for fp in files:
                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    face, bbox = detector.detect_and_crop(img)
                    if face is not None:
                        ok += 1
                total_ok += ok
                total_fail += len(files) - ok
                status = "OK" if ok > 0 else "FAIL"
                print(
                    f"  [{status:>4}] {src_name}/{cls}: "
                    f"detect được {ok}/{len(files)} face"
                )
    finally:
        detector.close()

    print("-" * 60)
    print(f"Tổng: detect được {total_ok}/{total_ok + total_fail} face")
    if total_fail > total_ok:
        print("[!] CẢNH BÁO: tỉ lệ detect thấp - có thể do MediaPipe không nhận được "
              "spoof image (in giấy / màn hình điện thoại). Đây là hiện tượng bình "
              "thường với spoof khó - sẽ bị skip lúc preprocess.")
    print("\nNếu mọi thứ OK → chạy: python scripts/01_prepare_data.py")


if __name__ == "__main__":
    main()
