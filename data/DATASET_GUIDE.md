# Dataset Guide - LCC-FASD

## Đã sử dụng: LCC-FASD

**Link:** https://www.kaggle.com/datasets/faber24/lcc-fasd

| Thuộc tính | Giá trị |
|---|---|
| Tên đầy đủ | Large Crowdcollected Facial Anti-Spoofing Dataset |
| Tác giả | Timoshenko et al. |
| Dung lượng | ~3 GB |
| Loại spoof | Print + Replay (điện thoại, màn hình) |

## Cấu trúc đặt vào `data/`

```
data/
├── LCC_FASD_training/         ← train (8.299 ảnh)
│   ├── real/   (1.223)
│   └── spoof/  (7.076)
├── LCC_FASD_development/      ← validation (2.948 ảnh)
│   ├── real/   (405)
│   └── spoof/  (2.543)
└── LCC_FASD_evaluation/       ← test (7.580 ảnh)
    ├── real/   (314)
    └── spoof/  (7.266)
```

Pipeline `scripts/01_prepare_data.py` sẽ:
1. Auto-detect cấu trúc trên.
2. Crop face (MediaPipe) + resize 224×224.
3. Lưu vào `data/processed/{train,val,test}/{real,spoof}/`.
4. **Giữ nguyên split** chính thức của tác giả → có thể so sánh trực tiếp với benchmark trong paper.

## Mức độ mất cân bằng (cần lưu ý)

| Split | Real | Spoof | Tỉ lệ |
|---|---|---|---|
| train | 1.223 | 7.076 | 1 : 5.8 |
| val | 405 | 2.543 | 1 : 6.3 |
| test | 314 | 7.266 | **1 : 23.1** ⚠ |

Pipeline đã xử lý:
- **Training:** dùng `class_weight` (sklearn-style 'balanced') để bù trừ.
- **Evaluation:** đọc FAR/FRR/EER/HTER thay vì Accuracy.

## Lưu ý đạo đức (cần ghi vào báo cáo)

- LCC-FASD chứa khuôn mặt người thật → tuân thủ giấy phép sử dụng cho mục đích academic.
- Không upload dataset lên public repo.
- Không dùng để training model nhận dạng danh tính cá nhân.
