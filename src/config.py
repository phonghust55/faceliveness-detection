"""Cấu hình tập trung cho toàn bộ dự án.

Tất cả hyperparameter và đường dẫn đặt ở đây để:
- Dễ thay đổi khi thử nghiệm
- Dễ liệt kê trong báo cáo (chỉ cần screenshot file này)
"""
import sys
from pathlib import Path

# Fix tiếng Việt có dấu trên Windows PowerShell (cp1252 → utf-8)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# === ĐƯỜNG DẪN ===
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Đảm bảo các thư mục output tồn tại
for _d in (CHECKPOINT_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# === DATASET ===
CLASS_NAMES = ["spoof", "real"]   # index 0 = spoof, index 1 = real (binary)
NUM_CLASSES = 1                   # Output 1 sigmoid → binary
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# === PREPROCESSING ===
IMG_SIZE = 224                    # MobileNetV2 / ResNet50 input chuẩn
FACE_MARGIN = 0.2                 # Mở rộng bbox 20% để giữ context (tóc, tai)
MIN_DETECTION_CONFIDENCE = 0.5

# === TRAINING ===
BATCH_SIZE = 32
EPOCHS_HEAD = 10                  # Stage 1: chỉ train head (freeze backbone)
EPOCHS_FINETUNE = 10              # Stage 2: unfreeze backbone, lr nhỏ
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5

# === EVALUATION ===
DECISION_THRESHOLD = 0.5          # Ngưỡng phân loại real/spoof
ROC_THRESHOLDS_NUM = 200          # Số ngưỡng để tính EER

# === WEBAPP ===
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = CHECKPOINT_DIR / "best_model.keras"
