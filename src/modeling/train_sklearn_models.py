"""
train_sklearn_models.py
=======================
Train scikit-learn Linear Regression and Random Forest Regression pipelines
for four water quality targets and save each as a .pkl file.

Output files (written to the same directory as this script):
    lr_water_temperature.pkl
    rf_water_temperature.pkl
    lr_ph.pkl
    rf_ph.pkl
    lr_dissolved_oxygen.pkl
    rf_dissolved_oxygen.pkl
    lr_nitrate.pkl
    rf_nitrate.pkl

Each .pkl is a fitted sklearn Pipeline whose feature order matches the
FEATURE_COLS list used by app.py at inference time.

Usage:
    python src/modeling/train_sklearn_models.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# PATHS  (relative to repo root — run from repo root)
# ─────────────────────────────────────────────────────────────
DATA_PATH = Path("data/tabular/merged/epa-climate-merged.csv")
OUTPUT_DIR = Path("src/modeling")
METRICS_PATH = Path("data/tabular/modeling/sklearn_model_metrics.csv")

# ─────────────────────────────────────────────────────────────
# FEATURES & TARGETS  (must match app.py exactly)
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "doy",
    "gdd_40_86",
    "high",
    "highc",
    "low",
    "lowc",
    "precip",
    "snow",
    "snowd",
    "distance_to_climate_station_km",
]

# display name → config
TARGETS = {
    "Water Temperature": {
        "column": "Temperature, water_value",
        "stem": "water_temperature",
        "valid_range": (-5.0, 45.0),
    },
    "pH": {
        "column": "pH_value",
        "stem": "ph",
        "valid_range": (0.0, 14.0),
    },
    "Dissolved Oxygen": {
        "column": "Dissolved oxygen (DO)_value",
        "stem": "dissolved_oxygen",
        "valid_range": (0.0, 30.0),
    },
    "Nitrate": {
        "column": "Nitrate_value",
        "stem": "nitrate",
        "valid_range": (0.0, 50.0),
    },
}

MODELS = {
    "Linear Regression": (
        "lr",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
    ),
    "Random Forest": (
        "rf",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    ),
}

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MIN_SAMPLES  = 50   # skip targets with fewer clean rows than this


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(safe) * 100.0)


def load_and_prepare(
    target_col: str,
    valid_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """
    Load the CSV, drop rows missing the target or any feature, impute remaining
    NaNs with column medians, and return (X, y) arrays.
    """
    df = pd.read_csv(DATA_PATH)

    needed = FEATURE_COLS + [target_col]
    df = df[needed].dropna(subset=[target_col])
    raw_rows = len(df)

    lower, upper = valid_range
    df = df[df[target_col].between(lower, upper)].copy()
    filtered_rows = len(df)

    if len(df) < MIN_SAMPLES:
        return None

    # Impute feature NaNs with column medians (mirrors app.py inference logic)
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y, raw_rows, filtered_rows


def train_and_save() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str]] = []

    for target_label, target_config in TARGETS.items():
        target_col = target_config["column"]
        stem = target_config["stem"]
        valid_range = target_config["valid_range"]
        print(f"\n--- {target_label} ({target_col}) ---")

        data = load_and_prepare(target_col, valid_range)
        if data is None:
            print(f"  [SKIP] fewer than {MIN_SAMPLES} complete rows — skipping.")
            continue

        X, y, raw_rows, filtered_rows = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(
            f"  Rows kept: {filtered_rows:,} of {raw_rows:,}"
            f"  |  train: {len(y_train):,}  test: {len(y_test):,}"
        )

        for model_label, (prefix, pipeline) in MODELS.items():
            print(f"  Training {model_label}...", end=" ", flush=True)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            error_rate = symmetric_mape(y_test, y_pred)
            print(
                f"R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"
                f"  Error={error_rate:.2f}%"
            )

            pkl_path = OUTPUT_DIR / f"{prefix}_{stem}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(pipeline, f)
            print(f"    Saved -> {pkl_path}")
            metrics_rows.append(
                {
                    "target": target_label,
                    "target_column": target_col,
                    "model": model_label,
                    "rows_before_filter": raw_rows,
                    "rows_used": filtered_rows,
                    "train_rows": len(y_train),
                    "test_rows": len(y_test),
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "error_rate_pct": error_rate,
                }
            )

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values(
            ["target", "r2"], ascending=[True, False]
        )
        metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"\nMetrics saved -> {METRICS_PATH}")
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_and_save()
