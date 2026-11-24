#!/bin/bash
# ShrimpMate AI Feeding Service - Curl Test Samples

BASE_URL="http://localhost:8000"

echo "=== 1. Health Check ==="
curl -s -X GET "${BASE_URL}/health" | jq .

echo -e "\n=== 2. XGBoost 7-Feature Feed Demand Prediction ==="
curl -s -X POST "${BASE_URL}/ai/predict-feed-demand" \
  -H "Content-Type: application/json" \
  -d '{
    "pondId": "208cf3b7-0000-4000-8000-000000000001",
    "doc": 45,
    "avgWeightG": 12.5,
    "doMin": 5.2,
    "deltaTan": 0.05,
    "morningRainMm": 0.0,
    "prevFeedAmountKg": 8.0,
    "waterChangeM3": 20.0
  }' | jq .
