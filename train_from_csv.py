"""
Script huấn luyện mô hình XGBoost Regressor từ Dataset CSV thực tế
Dành cho đề tài: ShrimpMate - Smart Shrimp Feeding System
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_NAMES = [
    "doc",
    "avg_weight_g",
    "do_min",
    "delta_tan",
    "morning_rain_mm",
    "prev_feed_kg",
    "water_change_m3",
]
TARGET_NAME = "feed_amount_kg"

def calculate_mape(y_true, y_pred, epsilon=1e-3):
    # Loại trừ các điểm cữ ăn bị chặn (y_true == 0) để tránh chia cho 0
    mask = y_true > epsilon
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def train_model(csv_path: str, output_model_path: str, output_meta_path: str):
    print("=" * 65)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH XGBOOST DỰ BÁO LƯỢNG THỨC ĂN TÔM")
    print(f"📂 Nguồn dữ liệu CSV: {csv_path}")
    print("=" * 65)

    if not os.path.exists(csv_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f" Tổng số bản ghi nạp được: {len(df):,} dòng")
    
    # Kiểm tra cột
    for col in FEATURE_NAMES + [TARGET_NAME]:
        if col not in df.columns:
            print(f"❌ LỖI: Thiếu cột bắt buộc '{col}' trong file CSV!")
            sys.exit(1)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[TARGET_NAME].values.astype(np.float32)

    # Chia tập dữ liệu 80% Train - 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f" Phân chia: Tập Train = {len(X_train):,} mẫu (80%) | Tập Test = {len(X_test):,} mẫu (20%)")

    # Khởi tạo mô hình XGBoost với siêu tham số chuẩn thực nghiệm
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    start_train_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_duration_sec = time.perf_counter() - start_train_time
    print(f"⚡ Thời gian huấn luyện: {train_duration_sec:.2f} giây")

    # Đánh giá trên tập kiểm định Test Set
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 0.0, None)

    mae = float(mean_absolute_error(y_test, test_preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    mape = calculate_mape(y_test, test_preds)
    r2 = float(r2_score(y_test, test_preds))

    # Đo độ trễ suy luận (Latency) trên 1 mẫu
    single_sample = X_test[:1]
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))

    # Tính độ quan trọng đặc trưng (Feature Importances)
    scores = model.feature_importances_
    total_score = sum(scores) if sum(scores) > 0 else 1.0
    feature_importances = {
        feat: float(round(score / total_score, 4))
        for feat, score in zip(FEATURE_NAMES, scores)
    }

    print("\n" + "=" * 65)
    print(" BẢNG KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (BENCHMARK METRICS):")
    print("=" * 65)
    print(f"| Chỉ số đánh giá                    | Giá trị thực nghiệm         |")
    print(f"|-------------------------------------|-----------------------------|")
    print(f"| Sai số tuyệt đối TB (MAE)          | {mae:.3f} kg/cữ              |")
    print(f"| Căn bậc 2 sai số bình phương (RMSE)| {rmse:.3f} kg/cữ              |")
    print(f"| Sai số phần trăm TB (MAPE)         | {mape:.2f} %                  |")
    print(f"| Hệ số xác định (R² Score)          | {r2:.3f}                      |")
    print(f"| Độ trễ suy luận đơn (Latency)      | {avg_latency_ms:.2f} ms                     |")
    print("=" * 65)

    print("\n📊 ĐỘ QUAN TRỌNG CỦA 7 ĐẶC TRƯNG ĐẦU VÀO:")
    for feat, score in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {feat:<20}: {score * 100:.2f}%")

    # Lưu weights mô hình
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save_model(output_model_path)
    print(f"\n[OK] Đã lưu mô hình XGBoost tại: {output_model_path}")

    # Lưu metadata
    meta = {
        "version": "v2.0.0",
        "datasetSource": os.path.basename(csv_path),
        "totalSamples": len(df),
        "trainSamples": len(X_train),
        "testSamples": len(X_test),
        "metrics": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape": round(mape, 2),
            "r2": round(r2, 3),
            "latencyMs": round(avg_latency_ms, 2),
        },
        "featureImportances": feature_importances,
        "lastRetrainedAt": datetime.now().isoformat(),
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Đã lưu thông tin Metadata tại: {output_meta_path}")
    print("=" * 65)
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN VÀ ĐÓNG GÓI HOÀN TẤT THÀNH CÔNG!")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình XGBoost từ file CSV")
    default_csv = os.path.join(os.path.dirname(__file__), "data", "shrimp_feeding_dataset.csv")
    default_model = os.path.join(os.path.dirname(__file__), "weights", "xgboost_feed_model.json")
    default_meta = os.path.join(os.path.dirname(__file__), "weights", "model_meta.json")

    parser.add_argument("--csv", type=str, default=default_csv, help="Đường dẫn tới file CSV dữ liệu")
    parser.add_argument("--model-out", type=str, default=default_model, help="Đường dẫn lưu model json")
    parser.add_argument("--meta-out", type=str, default=default_meta, help="Đường dẫn lưu metadata json")

    args = parser.parse_args()
    train_model(args.csv, args.model_out, args.meta_out)
