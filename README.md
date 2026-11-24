# ShrimpMate AI Service (Feeding Demand Prediction Microservice)

An independent microservice built with **Python (FastAPI)** designed for intelligent feeding demand prediction in the **ShrimpMate** industrial shrimp farming management system.

The system analyzes and predicts optimal feed allocation based on ecological factors, real-time water quality monitoring metrics, days of culture (DOC), and shrimp biomass.

---

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI, Uvicorn, Pydantic v2, Pydantic-settings
- **Machine Learning**:
  - `xgboost`: XGBoost Regressor model for feed amount prediction (kg/feeding session)
  - `scikit-learn`: Dataset splitting, model evaluation metrics (MAE, R²)
  - `numpy`: Agronomic data matrix processing
- **Deployment**: Docker, Docker Compose, Local Uvicorn

---

## Project Structure

```text
shrimpmate-AI/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app initialization and lifespan event handlers
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Pydantic-settings configuration
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── feed_prediction.py   # Feed prediction request & response schemas
│   │   └── system.py            # HealthResponse & ModelsLoadedStatus schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── xgboost_service.py   # XGBoost model service (7 environmental & growth features)
│   └── api/
│       ├── __init__.py
│       └── routes.py            # FastAPI route definitions
├── weights/                     # Serialized XGBoost model (xgboost_feed_model.json)
├── test_client.py               # Integration test script
└── README.md
```

---

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Run integration tests:
```bash
python test_client.py
```
