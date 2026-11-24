"""Integration test client for ShrimpMate AI Feeding Service."""

import json
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n--- 1. Testing GET /health ---")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {res.status_code}")
    print(json.dumps(res.json(), indent=2))
    assert res.status_code == 200
    assert res.json()["modelsLoaded"]["xgboost"] is True


def test_predict_feed_demand():
    print("\n--- 2. Testing POST /ai/predict-feed-demand ---")
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
    print(f"Status Code: {res.status_code}")
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    assert res.status_code == 200
    data = res.json()
    assert "predictedFeedAmountKg" in data
    assert "baseValueKg" in data
    assert len(data["featureContributions"]) == 7
    assert "explanation" in data


if __name__ == "__main__":
    print(f"Starting ShrimpMate AI Feeding Service test suite at {BASE_URL}...")
    try:
        test_health()
        test_predict_feed_demand()
        print("\nALL TEST SUITES PASSED SUCCESSFULLY!")
    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {BASE_URL}. Please start the server first (python -m uvicorn app.main:app --port 8000).")
    except Exception as e:
        print(f"\nTest execution failed: {e}")
