# ShrimpMate AI Service (Feeding Demand Prediction Microservice)

Microservice độc lập viết bằng **Python (FastAPI)** phục vụ dự báo định mức thức ăn thông minh cho hệ thống ao nuôi tôm công nghiệp **ShrimpMate**.

Hệ thống tự động cân nhắc và dự báo nhu cầu thức ăn tối ưu dựa trên các yếu tố sinh thái, chỉ số quan trắc chất lượng nước thời gian thực, ngày tuổi nuôi và sinh khối tôm.

---

## Tech Stack

- **Ngôn ngữ**: Python 3.10+
- **Framework**: FastAPI, Uvicorn, Pydantic v2, Pydantic-settings
- **Machine Learning**:
  - `xgboost`: Mô hình XGBoost Regressor dự báo khối lượng thức ăn (kg/cữ)
  - `scikit-learn`: Phân tách tập huấn luyện, đánh giá độ chính xác (MAE, R²)
  - `numpy`: Xử lý ma trận dữ liệu nông học
- **Triển khai**: Docker, Docker Compose, Local Uvicorn

---

## Cấu trúc thư mục

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

## Hướng dẫn khởi chạy

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
