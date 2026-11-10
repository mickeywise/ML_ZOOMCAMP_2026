
from fastapi.testclient import TestClient

from api.predict import app


client = TestClient(app)


# ---------------------------------------------------------
# 1. TEST HEALTH ENDPOINT
# ---------------------------------------------------------

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "airsense_random_forest_reproduced.joblib"


# ---------------------------------------------------------
# 2. TEST VALID PREDICTION
# ---------------------------------------------------------

def test_valid_prediction():

    payload = {
        "PM2_5": 37.0,
        "pm25_lag_1": 39.0,
        "pm25_lag_3": 45.0,
        "pm25_lag_6": 37.0,
        "pm25_lag_24": 128.0,
        "pm25_rolling_mean_3": 42.666667,
        "pm25_rolling_mean_6": 41.0,
        "pm25_rolling_mean_24": 116.75,
        "hour": 5,
        "day_of_week": 6,
        "month": 7,
        "day_of_year": 206,
        "TEMP": 24.7,
        "PRES": 999.7,
        "DEWP": 23.5,
        "RAIN": 0.0,
        "WSPM": 0.8,
        "station": "Aotizhongxin",
        "wd": "E"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_pm25" in data
    assert "forecast_horizon" in data
    assert "unit" in data

    assert isinstance(data["predicted_pm25"], float)
    assert data["predicted_pm25"] >= 0

    assert data["forecast_horizon"] == "1 hour ahead"
    assert data["unit"] == "µg/m³"


# ---------------------------------------------------------
# 3. TEST INVALID INPUT
# ---------------------------------------------------------

def test_invalid_prediction():

    payload = {
        "PM2_5": 37.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
