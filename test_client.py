"""Client kiểm thử tích hợp cho dịch vụ AI dự báo lượng thức ăn ShrimpMate."""

import json
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n--- 1. Kiểm tra GET /health ---")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Mã trạng thái phản hồi: {res.status_code}")
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    assert res.status_code == 200
    assert res.json()["modelsLoaded"]["xgboost"] is True


def test_predict_feed_demand():
    print("\n--- 2. Kiểm tra POST /ai/predict-feed-demand ---")
    payload = {
        "pondId": "208cf3b7-0000-4000-8000-000000000001",
        "doc": 45,
        "avgWeightG": 12.5,
        "doMin": 5.0,
        "deltaTan": 0.05,
        "morningRainMm": 0.0,
        "prevFeedAmountKg": 10.0,
        "waterChangeM3": 20.0,
    }
    res = requests.post(f"{BASE_URL}/ai/predict-feed-demand", json=payload)
    print(f"Mã trạng thái phản hồi: {res.status_code}")
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    assert res.status_code == 200
    data = res.json()
    assert "predictedFeedAmountKg" in data
    assert "baseValueKg" in data
    assert len(data["featureContributions"]) == 7
    assert "explanation" in data


if __name__ == "__main__":
    print(f"Khởi chạy bộ kiểm thử dịch vụ AI ShrimpMate tại {BASE_URL}...")
    try:
        test_health()
        test_predict_feed_demand()
        print("\nTẤT CẢ BỘ TEST ĐÃ HOÀN TẤT THÀNH CÔNG!")
    except requests.exceptions.ConnectionError:
        print(f"\nKhông thể kết nối đến {BASE_URL}. Vui lòng khởi động server trước (python -m uvicorn app.main:app --port 8000).")
    except Exception as e:
        print(f"\nKiểm thử thất bại: {e}")
