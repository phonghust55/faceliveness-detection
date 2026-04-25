# Face Liveness Detection (Anti-Spoofing)

Dự án môn **Xác thực Sinh trắc học (Biometrics)** - phân biệt khuôn mặt thật (Live) và khuôn mặt giả mạo qua ảnh/video (Spoof) bằng Deep Learning, có demo real-time qua webcam.

## 1. Mục tiêu

| Mục tiêu | Công cụ chính |
|---|---|
| Phát hiện khuôn mặt | MediaPipe Face Detection |
| Phân loại Live vs Spoof | MobileNetV2 (Transfer Learning) |
| Đánh giá theo chuẩn Biometrics | FAR, FRR, EER, HTER, ROC |
| Demo Real-time | FastAPI + WebRTC (`getUserMedia`) hoặc Streamlit |

## 2. Cài đặt môi trường

```bash
# Khuyến nghị Python 3.10
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Dataset

Dùng **LCC-FASD** (Large Crowdcollected Face Anti-Spoofing Dataset).

- **Tải:** https://www.kaggle.com/datasets/faber24/lcc-fasd
- **Đặt vào** thư mục `data/` với cấu trúc gốc của tác giả:

```
data/
├── LCC_FASD_training/{real,spoof}/        ← train set
├── LCC_FASD_development/{real,spoof}/     ← validation set
└── LCC_FASD_evaluation/{real,spoof}/      ← test set
```

Pipeline tự nhận diện cấu trúc này và **giữ nguyên split chính thức** → kết quả của bạn so sánh được với các benchmark đã công bố.

## 4. Pipeline chạy dự án

```bash
# Bước 1: Preprocessing - crop face + tổ chức train/val/test
python scripts/01_prepare_data.py

# Bước 2: Training (2-stage transfer learning, có class_weight cho imbalance)
python scripts/02_train.py

# Bước 3: Evaluation (sinh ROC, confusion matrix, FAR/FRR/EER/HTER)
python scripts/03_evaluate.py --weights checkpoints/best_model.keras

# Bước 4a: Chạy webapp (FastAPI)
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000
# Mở webapp/frontend/index.html bằng trình duyệt (hoặc Live Server VSCode)

# Bước 4b: Hoặc dùng Streamlit (gọn hơn cho demo)
streamlit run webapp/streamlit_app.py
```

## 5. Cấu trúc dự án

```
face-liveness-detection/
├── data/                      # Dataset (KHÔNG commit)
├── src/
│   ├── config.py              # Tập trung hyperparameter
│   ├── preprocessing/         # Face detection, augmentation, prepare_dataset
│   ├── models/                # MobileNetV2 + custom head
│   ├── training/              # Data loader + training loop
│   ├── evaluation/            # FAR/FRR/EER/HTER + visualization
│   └── inference/             # Predictor wrapper cho webapp
├── webapp/
│   ├── backend/               # FastAPI + Pydantic schemas
│   ├── frontend/              # HTML/JS vanilla + getUserMedia
│   └── streamlit_app.py       # Phương án B nhanh hơn
├── scripts/                   # CLI entrypoints
├── checkpoints/               # Model weights
└── results/                   # Plots + metrics report
```

## 6. Lưu ý về dataset LCC-FASD

LCC-FASD **mất cân bằng nặng**, đặc biệt ở test set:

| Split | Real | Spoof | Tỉ lệ |
|---|---|---|---|
| train (LCC_FASD_training) | 1.223 | 7.076 | 1 : 5.8 |
| val (LCC_FASD_development) | 405 | 2.543 | 1 : 6.3 |
| test (LCC_FASD_evaluation) | 314 | 7.266 | **1 : 23.1** |

→ **Hệ quả:** Accuracy có thể "ảo" (chỉ cần đoán toàn spoof là được ~96% trên test). Vì vậy:
- Code đã tự thêm `class_weight` khi training để loss công bằng giữa 2 class.
- **Khi đánh giá phải đọc FAR / FRR / EER / HTER**, không nhìn Accuracy đơn lẻ.

## 7. Kết quả tham chiếu

Theo benchmark của repo `kprokofi/light-weight-face-anti-spoofing` trên LCC-FASD:

| Model | EER (%) | ACER (%) |
|---|---|---|
| MN3_large | 16.13 | 15.40 |
| MN3_small | 18.70 | 19.69 |
| AENET | 20.91 | 22.61 |

→ Model MobileNetV2 trong dự án này nếu đạt **EER 15-20%** là hợp lý với scope sinh viên.
