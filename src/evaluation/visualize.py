"""Vẽ biểu đồ phục vụ báo cáo: training history, confusion matrix, ROC curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from ..config import PLOTS_DIR


def plot_history(history: dict, save_path: Path | None = None) -> Path:
    """Vẽ Loss & Accuracy theo epoch (gộp 2 stage)."""
    save_path = save_path or PLOTS_DIR / "training_history.png"

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["loss"], "o-", label="train_loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "s-", label="val_loss")
    axes[0].set_title("Loss qua các epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["accuracy"], "o-", label="train_acc")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], "s-", label="val_acc")
    axes[1].set_title("Accuracy qua các epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm: np.ndarray, save_path: Path | None = None) -> Path:
    """Heatmap confusion matrix với annotation."""
    save_path = save_path or PLOTS_DIR / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["spoof", "real"],
        yticklabels=["spoof", "real"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    eer: float,
    eer_threshold: float,
    save_path: Path | None = None,
) -> Path:
    """ROC curve + đánh dấu điểm EER (FAR = FRR).

    Đây là biểu đồ QUAN TRỌNG NHẤT cho báo cáo Biometrics: thể hiện sự đánh đổi
    giữa False Acceptance và False Rejection khi thay đổi ngưỡng quyết định.
    """
    save_path = save_path or PLOTS_DIR / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = np.trapz(tpr, fpr)
    fnr = 1 - tpr

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")

    # Đường y = 1 - x (FAR = FRR) cắt ROC tại EER
    ax.plot([0, 1], [1, 0], "r:", lw=1, alpha=0.6, label="FAR = FRR (EER line)")
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    ax.scatter(
        fpr[eer_idx], tpr[eer_idx], color="red", s=80, zorder=5,
        label=f"EER = {eer:.4f} @ thr={eer_threshold:.3f}",
    )

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 - FRR)")
    ax.set_title("ROC Curve - Face Anti-Spoofing")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
