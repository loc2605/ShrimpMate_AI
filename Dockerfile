# Dockerfile đa giai đoạn cho ShrimpMate AI Feeding Service (Mô hình XGBoost bảng nhẹ)
FROM python:3.10-slim AS builder

WORKDIR /app

# Cài đặt các gói phụ thuộc để build hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt


# Giai đoạn chạy thực tế (Production image)
FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện runtime tối thiểu (OpenMP cho XGBoost + curl để kiểm tra healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép các gói Python đã cài đặt từ builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Ho_Chi_Minh

# Sao chép mã nguồn và thư mục trọng số mô hình
COPY app/ /app/app/
COPY weights/ /app/weights/
COPY .env.example /app/.env

# Cổng mặc định
EXPOSE 8000

# Kiểm tra trạng thái hoạt động
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Khởi chạy máy chủ FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
