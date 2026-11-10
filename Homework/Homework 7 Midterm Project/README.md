# AirSense: One-Hour-Ahead PM2.5 Forecasting

## 1. Project Overview

AirSense is a machine learning project that forecasts hourly PM2.5 (fine particulate matter) concentrations using historical air quality measurements and meteorological observations.

The project uses the Beijing Multi-Site Air Quality dataset to train a Random Forest regression model that predicts PM2.5 concentrations one hour ahead.

The objective is to develop an end-to-end machine learning application covering data preparation, exploratory data analysis, feature engineering, model training, evaluation, API development, automated testing and containerisation.

**Project type:** Supervised Machine Learning – Regression  
**Prediction target:** PM2.5 concentration one hour ahead  
**Model:** Random Forest Regressor  
**Deployment framework:** FastAPI  
**Containerisation:** Docker  
**Prediction unit:** µg/m³

## 2. Problem Description

Air pollution, particularly fine particulate matter (PM2.5), is an important environmental concern.

PM2.5 consists of particles with an aerodynamic diameter of 2.5 micrometres or smaller. Exposure to elevated concentrations can have adverse effects on human health.

AirSense explores whether historical pollution measurements, temporal patterns and meteorological variables can be used to forecast short-term PM2.5 concentrations.

The model uses observations available at the current hour to estimate the PM2.5 concentration at the next hour.

This is an experimental forecasting application and is not intended to provide official air quality measurements, public health alerts or medical advice.

## 3. Dataset

The project uses the **Beijing Multi-Site Air Quality Data Set** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/501/beijing
- Data collection period: March 2013 – February 2017
- Number of monitoring stations: 12
- Total observations: 420,768 hourly records
- Number of original variables: 18
- License: CC BY 4.0

The dataset contains six air quality measurements and six meteorological variables, alongside date, time and monitoring station information.

### Target variable

`PM2.5` – Hourly concentration of fine particulate matter.

### Input variables

The model uses:

- Historical PM2.5 measurements and lag features.
- Rolling averages over previous 3, 6 and 24 hours.
- Calendar features, including hour, day of the week, month and day of the year.
- Meteorological variables, including temperature, pressure, dew point, rainfall and wind speed.
- Wind direction and monitoring station.

Missing predictor values are handled through preprocessing pipelines. Rows without a valid target value are excluded from model training and evaluation.

## 4. Data Preparation and Feature Engineering

The original dataset consists of individual CSV files for each monitoring station. These files are combined into a single dataset.

The following preparation steps are performed:

1. Combine the 12 station datasets.
2. Construct a datetime variable from the year, month, day and hour columns.
3. Sort observations by monitoring station and timestamp.
4. Generate the one-hour-ahead prediction target.
5. Create historical lag features.
6. Calculate rolling averages using previous observations only.
7. Extract calendar features.
8. Separate the dataset chronologically into training, validation and test sets.

### Target construction

The prediction target is generated using the next hourly PM2.5 observation within each monitoring station.

### Leakage prevention

Lag and rolling features are calculated separately within each station. Rolling averages are shifted to ensure that the current target observation does not contribute to its own predictors.

The dataset is divided chronologically rather than randomly to preserve the forecasting scenario.

The split boundaries are based on the target timestamp to prevent observations with overlapping target periods from crossing dataset boundaries.

| Dataset | Observations |
|---|---:|
| Training | 288,246 |
| Validation | 61,785 |
| Testing | 61,986 |
| Total | 412,017 |

## 5. Model Development

Several regression algorithms were evaluated during model development:

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

The Random Forest model performed best among the initial candidates.

Hyperparameter optimisation was conducted using RandomizedSearchCV with expanding chronological validation folds.

### Selected model

**Random Forest Regressor**

| Hyperparameter | Selected value |
|---|---:|
| n_estimators | 150 |
| max_depth | None |
| min_samples_leaf | 10 |
| max_features | 0.7 |
| random_state | 42 |

The preprocessing pipeline uses median imputation and standardisation for numerical features, and most-frequent imputation and one-hot encoding for categorical features.

## 6. Model Evaluation

The final model was evaluated using the untouched chronological test set.

| Metric | Test result |
|---|---:|
| Mean Absolute Error (MAE) | 10.227 |
| Root Mean Squared Error (RMSE) | 20.085 |
| R² Score | 0.953 |

### Metric interpretation

- **MAE:** The average absolute prediction error is approximately 10.23 µg/m³.
- **RMSE:** The root mean squared prediction error is approximately 20.09 µg/m³. This metric penalises larger errors more heavily.
- **R²:** The model explains approximately 95.3% of the variance in the test target values.

Model performance is evaluated on the same 12 Beijing monitoring stations represented in the training dataset. These results should not be interpreted as evidence of generalisation to other cities or monitoring networks.

## 7. Project Structure

```text
Homework 7 Midterm Project/
│
├── api/
│   └── predict.py
│
├── data/
│   ├── raw/
│   │   └── beijing_air_quality/
│   └── processed/
│
├── models/
│   ├── airsense_random_forest.joblib
│   └── airsense_random_forest_reproduced.joblib
│
├── notebooks/
│   └── notebook.ipynb
│
├── reports/
│   └── figures/
│       └── actual_vs_predicted_pm25.png
│
├── src/
│   ├── download_data.py
│   ├── inspect_data.py
│   └── train.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## 8. Installation and Local Setup

### Prerequisites

- Python
- pip
- Git (optional)
- Docker Desktop (for containerised execution)

### Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Create a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Download and prepare the dataset

Follow the instructions in `src/download_data.py` and ensure the original dataset is available under:

```text
data/raw/beijing_air_quality/
```

### Train the model

```bash
python src/train.py
```

The training script evaluates the model and exports the trained pipeline to:

```text
models/airsense_random_forest_reproduced.joblib
```

## 9. Running the API Locally

Start the FastAPI application:

```bash
python -m uvicorn api.predict:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model availability |
| POST | `/predict` | Generates a one-hour-ahead PM2.5 prediction |

### Example health response

```json
{
  "status": "healthy",
  "model": "airsense_random_forest_reproduced.joblib"
}
```

### Example prediction request

```json
{
  "PM2_5": 37,
  "pm25_lag_1": 39,
  "pm25_lag_3": 45,
  "pm25_lag_6": 37,
  "pm25_lag_24": 128,
  "pm25_rolling_mean_3": 42.666667,
  "pm25_rolling_mean_6": 41,
  "pm25_rolling_mean_24": 116.75,
  "hour": 5,
  "day_of_week": 6,
  "month": 7,
  "day_of_year": 206,
  "TEMP": 24.7,
  "PRES": 999.7,
  "DEWP": 23.5,
  "RAIN": 0,
  "WSPM": 0.8,
  "station": "Aotizhongxin",
  "wd": "E"
}
```

### Example prediction response

```json
{
  "predicted_pm25": 41.749,
  "forecast_horizon": "1 hour ahead",
  "unit": "µg/m³"
}
```

**Important:** The API expects pre-engineered features. The caller must calculate the historical lag and rolling-average variables before submitting a prediction request.

## 10. Automated Testing

The API is tested using pytest and FastAPI's TestClient.

Run the tests:

```bash
python -m pytest tests/test_api.py -v
```

Current test results:

- Health endpoint: PASSED
- Valid prediction: PASSED
- Invalid prediction: PASSED

**Latest result: 3 passed.**

## 11. Docker Deployment

AirSense can be run as a Docker container without requiring the local Python virtual environment.

### Build the image

```bash
docker build -t airsense-api:1.0 .
```

### Run the container

```bash
docker run --rm -d \
  --name airsense-container \
  -p 8000:8000 \
  airsense-api:1.0
```

For Windows Command Prompt, the same command can be written on one line:

```cmd
docker run --rm -d --name airsense-container -p 8000:8000 airsense-api:1.0
```

### Verify the running container

```bash
docker ps
```

### Check container logs

```bash
docker logs airsense-container
```

### Test the API

Health endpoint:

```text
http://localhost:8000/health
```

Interactive documentation:

```text
http://localhost:8000/docs
```

### Stop the container

```bash
docker stop airsense-container
```

## 12. Limitations and Future Improvements

Current limitations:

- The model was developed using historical observations from Beijing and has not been validated on other geographical locations.
- The API requires callers to supply pre-engineered historical features.
- Weather predictors represent current-hour observations, not future weather forecasts.
- Prediction errors increase at the highest PM2.5 concentrations.
- The model is a prototype and should not be used as an official air quality warning system.

Potential improvements:

1. Develop a feature-engineering service that automatically calculates lag and rolling features.
2. Integrate external weather forecast data.
3. Evaluate additional forecasting algorithms, including gradient boosting models.
4. Improve prediction accuracy during extreme pollution events.
5. Implement cloud deployment and expose a publicly accessible API.
6. Add monitoring, logging and model performance tracking.

## 13. Acknowledgements

Dataset:

Zhang, S., Guo, B., Dong, A., He, J., Xu, Z. and Chen, S. X. (2017). *Cautionary Tales on Air-Quality Improvement in Beijing*. Proceedings of the Royal Society A.

UCI Machine Learning Repository: Beijing Multi-Site Air Quality Data Set.

Dataset licence: CC BY 4.0.

---

**Project:** AirSense  
**Focus:** Machine Learning | Regression | Time-Series Forecasting | FastAPI | Docker  
**Status:** Model trained, API tested and Docker container operational.
