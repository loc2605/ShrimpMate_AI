# ShrimpMate AI Service (Feeding Demand Prediction Microservice)

Microservice độc lập viết bằng **Python (FastAPI)** phục vụ dự báo định mức thức ăn thông minh cho hệ thống ao nuôi tôm công nghiệp **ShrimpMate**.

Hệ thống tự động cân nhắc và dự báo nhu cầu thức ăn tối ưu dựa trên các yếu tố sinh thái, chỉ số quan trắc chất lượng nước thời gian thực, ngày tuổi nuôi và sinh khối tôm.

---

## 🛠 Tech Stack

- **Ngôn ngữ**: Python 3.10+
- **Framework**: FastAPI, Uvicorn, Pydantic v2, Pydantic-settings
- **Machine Learning**:
  - `xgboost`: Mô hình XGBoost Regressor dự báo khối lượng thức ăn (kg/cữ)
  - `scikit-learn`: Phân tách tập huấn luyện, đánh giá độ chính xác (MAE, R²)
  - `numpy`: Xử lý ma trận dữ liệu nông học
- **Triển khai**: Docker, Docker Compose, Local Uvicorn

---

## 📁 Cấu trúc thư mục

```text
shrimpmate-AI/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entrypoint khởi tạo FastAPI app, lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Cấu hình Pydantic-settings
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── feed_prediction.py   # Schema Request & Result dự báo thức ăn
│   │   └── system.py            # Schema HealthResponse & ModelsLoadedStatus
│   ├── services/
│   │   ├── __init__.py
│   │   └── xgboost_service.py   # Xử lý mô hình XGBoost (7 đặc trưng môi trường & tăng trưởng)
│   └── api/
│       ├── __init__.py
│       └── routes.py            # Định nghĩa các router FastAPI
├── weights/                     # Chứa mô hình XGBoost (xgboost_feed_model.json)
├── test_client.py               # Script kiểm thử tích hợp
└── README.md
```

---

## 📡 API Endpoints

### 1. Dự báo lượng thức ăn (XGBoost Regressor)
- **Method**: `POST`
- **Path**: `/ai/predict-feed-demand`
- **Request Body** (`FeedPredictionRequest`):
```json
{
  "pondId": "208cf3b7-0000-4000-8000-000000000001",
  "doc": 45,
  "avgWeightG": 12.5,
  "doMin": 5.2,
  "deltaTan": 0.05,
  "morningRainMm": 0.0,
  "prevFeedAmountKg": 8.0,
  "waterChangeM3": 20.0
}
```

- **Response Body** (`FeedPredictionResult`):
```json
{
  "predictedFeedAmountKg": 10.45,
  "baseValueKg": 7.62,
  "modelName": "XGBoost-Regressor",
  "modelVersion": "v2.0.0",
  "featureImportance": {
    "doc": 0.0576,
    "avg_weight_g": 0.2848,
    "do_min": 0.1997,
    "delta_tan": 0.0237,
    "morning_rain_mm": 0.2506,
    "prev_feed_kg": 0.1701,
    "water_change_m3": 0.0135
  },
  "featureContributions": [
    {
      "feature": "avg_weight_g",
      "name": "Trọng lượng bình quân (ABW)",
      "value": "12.50 g/con",
      "impact": "INCREASE",
      "deltaKg": 1.12,
      "description": "Trọng lượng bình quân (ABW) (12.50 g/con) làm tăng dự báo khẩu phần +1.12 kg theo mô hình XGBoost (TreeSHAP)."
    }
  ],
  "explanation": "Tôm DOC 45 ngày (ABW 12.50 g/con), lượng ăn cữ trước 8.00 kg. Khuyến nghị XGBoost cấp phát: 10.45 kg cho cữ ăn này."
}
```

---

### 2. Health check
- **Method**: `GET`
- **Path**: `/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "ShrimpMate AI Feeding Service",
  "version": "2.0.0",
  "modelName": "XGBoost-Regressor",
  "modelsLoaded": {
    "xgboost": true
  }
}
```

---

## 🚀 Hướng dẫn khởi chạy

1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

2. Khởi chạy server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Chạy test tích hợp:
```bash
python test_client.py
```
