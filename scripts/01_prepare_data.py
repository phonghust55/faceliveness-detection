"""CLI: chuẩn bị dataset (crop face + tổ chức train/val/test).

Tự động phát hiện cấu trúc dataset:
  - Nếu thấy data/LCC_FASD_{training,development,evaluation}/ → giữ split gốc.
  - Nếu thấy data/raw/{real,spoof}/                          → tự chia 70/15/15.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --no_clean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.preprocessing.prepare_dataset import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Face Anti-Spoofing dataset")
    parser.add_argument("--out_dir", type=Path, default=PROCESSED_DIR,
                        help="Thư mục output sau preprocessing")
    parser.add_argument("--no_clean", action="store_true",
                        help="Không xoá thư mục output cũ trước khi chạy")
    args = parser.parse_args()

    run(out_dir=args.out_dir, clean=not args.no_clean)


if __name__ == "__main__":
    main()
