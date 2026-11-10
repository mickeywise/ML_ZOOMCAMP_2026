
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "airsense_random_forest_reproduced.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# 2. DEFINE THE INPUT SCHEMA
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False
    )

    # Current PM2.5 and historical readings
    PM2_5: float
    pm25_lag_1: float
    pm25_lag_3: float
    pm25_lag_6: float
    pm25_lag_24: float

    # Historical rolling averages
    pm25_rolling_mean_3: float
    pm25_rolling_mean_6: float
    pm25_rolling_mean_24: float

    # Calendar features
    hour: int
    day_of_week: int
    month: int
    day_of_year: int

    # Current-hour weather observations
    TEMP: float
    PRES: float
    DEWP: float
    RAIN: float
    WSPM: float

    # Categorical features
    station: str
    wd: str


# ---------------------------------------------------------
# 3. INITIALISE FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="AirSense PM2.5 Forecasting API",
    description=(
        "Predicts PM2.5 concentration one hour ahead "
        "using current pollution, historical readings "
        "and weather observations."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# 4. HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": MODEL_PATH.name
    }


# ---------------------------------------------------------
# 5. PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert the validated request into a DataFrame.
    input_data = request.model_dump()

    # Match the original training column name.
    input_data["PM2.5"] = input_data.pop("PM2_5")

    input_df = pd.DataFrame([input_data])

    # Generate the one-hour-ahead prediction.
    prediction = model.predict(input_df)[0]

    return {
        "predicted_pm25": round(float(prediction), 3),
        "forecast_horizon": "1 hour ahead",
        "unit": "µg/m³"
    }
