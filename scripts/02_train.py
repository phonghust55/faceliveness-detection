"""CLI: train model 2 stage và lưu best checkpoint.

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --resume   # bỏ qua Stage 1, dùng best_model_head.keras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.visualize import plot_history
from src.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Face Anti-Spoofing model")
    parser.add_argument(
        "--resume", action="store_true",
        help="Bỏ qua Stage 1, load checkpoints/best_model_head.keras và "
             "chỉ chạy Stage 2 (fine-tune)",
    )
    args = parser.parse_args()

    model, history = train(skip_stage1=args.resume)
    if history:
        plot_path = plot_history(history)
        print(f"\n[+] Biểu đồ training history đã lưu tại: {plot_path}")


if __name__ == "__main__":
    main()
