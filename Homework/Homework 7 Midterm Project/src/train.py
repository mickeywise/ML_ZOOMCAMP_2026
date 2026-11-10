
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "beijing_air_quality"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "airsense_random_forest_reproduced.joblib"

RANDOM_STATE = 42

# These are the exact target-date boundaries used in the notebook.
TRAIN_END = pd.Timestamp("2015-12-18 05:00:00")
VAL_END = pd.Timestamp("2016-07-24 05:00:00")


# ---------------------------------------------------------
# 2. LOAD THE DATASET
# ---------------------------------------------------------

def load_data():
    csv_files = sorted(DATA_DIR.rglob("PRSA_Data_*.csv"))

    if len(csv_files) != 12:
        raise ValueError(
            f"Expected 12 station CSV files, but found {len(csv_files)} "
            f"under {DATA_DIR}"
        )

    dataframes = [pd.read_csv(file) for file in csv_files]

    df = pd.concat(dataframes, ignore_index=True)

    print(f"Loaded {len(csv_files)} station files.")
    print(f"Combined dataset shape: {df.shape}")
    print(f"Stations: {df['station'].nunique()}")

    return df


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(df):
    df = df.copy()

    # Construct timestamp and sort chronologically by station.
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
    )

    df = df.sort_values(
        ["station", "datetime"]
    ).reset_index(drop=True)

    # Target: PM2.5 one hour into the future, within each station.
    df["target_pm25"] = (
        df.groupby("station")["PM2.5"].shift(-1)
    )

    # Historical PM2.5 lags.
    for lag in [1, 3, 6, 24]:
        df[f"pm25_lag_{lag}"] = (
            df.groupby("station")["PM2.5"].shift(lag)
        )

    # Rolling averages using previous observations only.
    for window in [3, 6, 24]:
        df[f"pm25_rolling_mean_{window}"] = (
            df.groupby("station")["PM2.5"]
            .transform(
                lambda x: x.shift(1).rolling(
                    window=window,
                    min_periods=window
                ).mean()
            )
        )

    # Calendar features.
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_of_week
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.day_of_year

    # Timestamp corresponding to the prediction target.
    df["target_datetime"] = (
        df["datetime"] + pd.Timedelta(hours=1)
    )

    # Remove rows without an observed target.
    model_df = df.dropna(
        subset=["target_pm25"]
    ).copy()

    return model_df


# ---------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ---------------------------------------------------------

def prepare_features(model_df):

    numeric_features = [
        "PM2.5",
        "pm25_lag_1",
        "pm25_lag_3",
        "pm25_lag_6",
        "pm25_lag_24",
        "pm25_rolling_mean_3",
        "pm25_rolling_mean_6",
        "pm25_rolling_mean_24",
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM"
    ]

    categorical_features = [
        "station",
        "wd"
    ]

    features = numeric_features + categorical_features
    target = "target_pm25"

    missing_columns = [
        col for col in features + [target]
        if col not in model_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return numeric_features, categorical_features, features, target


# ---------------------------------------------------------
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------

def split_data(model_df, features, target):

    train_df = model_df[
        model_df["target_datetime"] <= TRAIN_END
    ].copy()

    val_df = model_df[
        (model_df["target_datetime"] > TRAIN_END)
        & (model_df["target_datetime"] <= VAL_END)
    ].copy()

    test_df = model_df[
        model_df["target_datetime"] > VAL_END
    ].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("One or more data splits are empty.")

    # Verify that the target periods do not overlap.
    assert train_df["target_datetime"].max() < (
        val_df["target_datetime"].min()
    )

    assert val_df["target_datetime"].max() < (
        test_df["target_datetime"].min()
    )

    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]

    y_train = train_df[target]
    y_val = val_df[target]
    y_test = test_df[target]

    print("\nSplit sizes:")
    print(f"Training:   {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    print("\nTarget date boundaries:")
    print(
        "Training:",
        train_df["target_datetime"].min(),
        "to",
        train_df["target_datetime"].max()
    )
    print(
        "Validation:",
        val_df["target_datetime"].min(),
        "to",
        val_df["target_datetime"].max()
    )
    print(
        "Test:",
        test_df["target_datetime"].min(),
        "to",
        test_df["target_datetime"].max()
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------
# 6. BUILD THE PREPROCESSING PIPELINE
# ---------------------------------------------------------

def create_preprocessor(
    numeric_features,
    categorical_features
):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# 7. TRAIN THE TUNED RANDOM FOREST
# ---------------------------------------------------------

def train_model(
    numeric_features,
    categorical_features,
    X_train,
    y_train
):

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=10,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    return pipeline


# ---------------------------------------------------------
# 8. EVALUATE THE MODEL
# ---------------------------------------------------------

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(y, predictions)

    print(f"\n{dataset_name} results:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ---------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AIRSENSE - PM2.5 FORECASTING MODEL")
    print("=" * 60)

    df = load_data()

    model_df = create_features(df)

    print(f"\nModel dataset shape: {model_df.shape}")

    (
        numeric_features,
        categorical_features,
        features,
        target
    ) = prepare_features(model_df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(model_df, features, target)

    model = train_model(
        numeric_features,
        categorical_features,
        X_train,
        y_train
    )

    # Evaluate all three partitions.
    evaluate_model(
        model,
        X_train,
        y_train,
        "Training"
    )

    evaluate_model(
        model,
        X_val,
        y_val,
        "Validation"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    # Save the independently reproduced pipeline.
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
