"""Metrics chuyên ngành Biometrics cho Face Anti-Spoofing.

Quy ước nhãn:
    - y_true = 1  →  Real (Genuine / Live)
    - y_true = 0  →  Spoof (Imposter / Fake)
    - y_score    →  xác suất sigmoid model dự đoán "real"
    - threshold  →  ngưỡng quyết định (mặc định 0.5)

Định nghĩa (theo ISO/IEC 30107-3 - chuẩn quốc tế cho PAD):

    FAR  (False Acceptance Rate)
        = APCER (Attack Presentation Classification Error Rate)
        = số ảnh SPOOF bị nhận nhầm là REAL  /  tổng ảnh SPOOF
        → Càng thấp càng AN TOÀN (kẻ giả mạo khó vượt qua).

    FRR  (False Rejection Rate)
        = BPCER (Bona-fide Presentation Classification Error Rate)
        = số ảnh REAL bị từ chối thành SPOOF  /  tổng ảnh REAL
        → Càng thấp càng TIỆN DỤNG (người thật ít bị làm phiền).

    EER  (Equal Error Rate)
        = giá trị tại điểm FAR(t) ≈ FRR(t)
        → Single-number metric tốt nhất khi muốn so sánh các hệ thống biometric.

    HTER (Half Total Error Rate)
        = (FAR + FRR) / 2  tại một ngưỡng cố định
        → Phổ biến trong các paper về PAD (CASIA-FASD, OULU-NPU dùng HTER).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


@dataclass
class BiometricMetrics:
    """Tập hợp các chỉ số đánh giá."""
    accuracy: float
    far: float            # tại ngưỡng cho trước
    frr: float            # tại ngưỡng cho trước
    hter: float           # = (FAR + FRR) / 2
    eer: float            # Equal Error Rate
    eer_threshold: float  # ngưỡng tại đó FAR = FRR
    auc: float
    threshold: float      # ngưỡng được dùng cho FAR/FRR/HTER
    confusion: np.ndarray
    classification_text: str


def compute_far_frr(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    """Tính FAR và FRR tại 1 ngưỡng cụ thể.

    y_true ∈ {0=spoof, 1=real},  y_score ∈ [0, 1] (xác suất real).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_score).ravel() >= threshold).astype(int)

    # Số ảnh spoof bị dự đoán nhầm là real
    fa = np.sum((y_true == 0) & (y_pred == 1))
    n_spoof = np.sum(y_true == 0)
    far = fa / n_spoof if n_spoof > 0 else 0.0

    # Số ảnh real bị dự đoán nhầm là spoof
    fr = np.sum((y_true == 1) & (y_pred == 0))
    n_real = np.sum(y_true == 1)
    frr = fr / n_real if n_real > 0 else 0.0
    return float(far), float(frr)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate và ngưỡng tương ứng.

    Cách tính: vẽ FAR(t) và FRR(t) theo t, tìm t* sao cho |FAR - FRR| nhỏ nhất.
    Dùng `roc_curve` của sklearn để có sẵn fpr=FAR, tpr=TPR (1-FRR).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr                       # FRR = 1 - TPR (vì lớp positive là "real")
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> BiometricMetrics:
    """Tổng hợp toàn bộ metrics cần cho báo cáo Biometrics."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(int)

    far, frr = compute_far_frr(y_true, y_score, threshold)
    eer, eer_thr = compute_eer(y_true, y_score)

    # AUC từ ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(np.trapz(tpr, fpr))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, target_names=["spoof", "real"], digits=4
    )

    return BiometricMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        far=far,
        frr=frr,
        hter=(far + frr) / 2,
        eer=eer,
        eer_threshold=eer_thr,
        auc=auc,
        threshold=threshold,
        confusion=cm,
        classification_text=report,
    )


def format_report(m: BiometricMetrics) -> str:
    """Định dạng metric để in ra console / lưu file."""
    lines = [
        "=" * 60,
        "BIOMETRIC EVALUATION REPORT",
        "=" * 60,
        f"  Threshold        : {m.threshold:.4f}",
        f"  Accuracy         : {m.accuracy:.4f}",
        f"  AUC (ROC)        : {m.auc:.4f}",
        "-" * 60,
        f"  FAR  @ thr={m.threshold:.2f} : {m.far:.4f}   (spoof bị nhận thành real)",
        f"  FRR  @ thr={m.threshold:.2f} : {m.frr:.4f}   (real bị từ chối thành spoof)",
        f"  HTER @ thr={m.threshold:.2f} : {m.hter:.4f}",
        f"  EER              : {m.eer:.4f}   (tại ngưỡng {m.eer_threshold:.4f})",
        "-" * 60,
        "Confusion Matrix [rows=true, cols=pred]:",
        f"             pred=spoof  pred=real",
        f"  true=spoof   {m.confusion[0,0]:>8d}  {m.confusion[0,1]:>8d}",
        f"  true=real    {m.confusion[1,0]:>8d}  {m.confusion[1,1]:>8d}",
        "-" * 60,
        m.classification_text,
        "=" * 60,
    ]
    return "\n".join(lines)
