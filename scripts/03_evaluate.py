"""CLI: chạy evaluation trên test set, sinh ROC + Confusion Matrix + báo cáo.

Usage:
    python scripts/03_evaluate.py
    python scripts/03_evaluate.py --weights checkpoints/best_model.keras
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DECISION_THRESHOLD, MODEL_PATH, REPORTS_DIR
from src.evaluation.metrics import evaluate, format_report
from src.evaluation.visualize import plot_confusion_matrix, plot_roc_curve
from src.models.loader import load_liveness_model
from src.training.data_loader import get_datasets


def collect_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_all, y_score_all = [], []
    for x, y in test_ds:
        y_score_all.append(model.predict(x, verbose=0).ravel())
        y_true_all.append(y.numpy().ravel())
    return np.concatenate(y_true_all), np.concatenate(y_score_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Face Anti-Spoofing model")
    parser.add_argument("--weights", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Model không tồn tại: {args.weights}")

    print(f"[*] Loading model from {args.weights}")
    model = load_liveness_model(args.weights)

    _, _, test_ds = get_datasets()
    print("[*] Đang predict trên test set ...")
    y_true, y_score = collect_predictions(model, test_ds)
    print(f"[+] Tổng số mẫu test: {len(y_true)}  (real={int(y_true.sum())}, "
          f"spoof={int((1 - y_true).sum())})")

    # Tính metrics
    metrics = evaluate(y_true, y_score, threshold=args.threshold)
    report_text = format_report(metrics)
    print("\n" + report_text)

    # Lưu reports
    txt_path = REPORTS_DIR / "evaluation_report.txt"
    json_path = REPORTS_DIR / "metrics.json"
    txt_path.write_text(report_text, encoding="utf-8")
    metrics_dict = asdict(metrics)
    metrics_dict["confusion"] = metrics.confusion.tolist()
    metrics_dict.pop("classification_text", None)
    json_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    print(f"\n[+] Đã lưu báo cáo: {txt_path}")
    print(f"[+] Đã lưu metrics JSON: {json_path}")

    # Vẽ
    cm_path = plot_confusion_matrix(metrics.confusion)
    roc_path = plot_roc_curve(y_true, y_score, metrics.eer, metrics.eer_threshold)
    print(f"[+] Confusion matrix: {cm_path}")
    print(f"[+] ROC curve:        {roc_path}")


if __name__ == "__main__":
    main()
