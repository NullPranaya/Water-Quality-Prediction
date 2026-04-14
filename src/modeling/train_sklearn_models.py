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
DATA_PATH  = Path("data/tabular/merged/epa-climate-merged.csv")
OUTPUT_DIR = Path("src/modeling")   # same folder as this script

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

# display name → (csv column, filename stem)
TARGETS = {
    "Water Temperature": ("Temperature, water_value",   "water_temperature"),
    "pH":                ("pH_value",                   "ph"),
    "Dissolved Oxygen":  ("Dissolved oxygen (DO)_value","dissolved_oxygen"),
    "Nitrate":           ("Nitrate_value",               "nitrate"),
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


def load_and_prepare(target_col: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the CSV, drop rows missing the target or any feature, impute remaining
    NaNs with column medians, and return (X, y) arrays.
    """
    df = pd.read_csv(DATA_PATH)

    # Keep only rows that have both the target and all feature values
    needed = FEATURE_COLS + [target_col]
    df = df[needed].dropna(subset=[target_col])

    if len(df) < MIN_SAMPLES:
        return None

    # Impute feature NaNs with column medians (mirrors app.py inference logic)
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def train_and_save() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for target_label, (target_col, stem) in TARGETS.items():
        print(f"\n--- {target_label} ({target_col}) ---")

        data = load_and_prepare(target_col)
        if data is None:
            print(f"  [SKIP] fewer than {MIN_SAMPLES} complete rows — skipping.")
            continue

        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"  Rows: {len(y):,}  |  train: {len(y_train):,}  test: {len(y_test):,}")

        for model_label, (prefix, pipeline) in MODELS.items():
            print(f"  Training {model_label}...", end=" ", flush=True)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            rmse   = mean_squared_error(y_test, y_pred) ** 0.5
            mae    = mean_absolute_error(y_test, y_pred)
            r2     = r2_score(y_test, y_pred)
            print(f"R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

            pkl_path = OUTPUT_DIR / f"{prefix}_{stem}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(pipeline, f)
            print(f"    Saved -> {pkl_path}")

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_and_save()
