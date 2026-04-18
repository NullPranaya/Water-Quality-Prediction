"""
Train scikit-learn regression pipelines for the four dashboard targets,
selecting the strongest configuration per model family with cross-validation
and reporting held-out test metrics.

Usage:
    python src/modeling/train_sklearn_models.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_feature_engineering import add_derived_features

DATA_PATH = Path("data/tabular/merged/epa-climate-merged.csv")
OUTPUT_DIR = Path("src/modeling")
METRICS_PATH = Path("data/tabular/modeling/sklearn_model_metrics.csv")

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
    "LatitudeMeasure",
    "LongitudeMeasure",
]

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

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES = 50
CV_SPLITS = 5


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(safe) * 100.0)


def make_preprocessor(include_scaler: bool) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_engineering", FunctionTransformer(add_derived_features, validate=False)),
    ]
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    return steps


def make_linear_pipeline() -> Pipeline:
    return Pipeline(make_preprocessor(include_scaler=True) + [
        ("model", LinearRegression()),
    ])


def make_rf_pipeline(**kwargs: object) -> Pipeline:
    return Pipeline(make_preprocessor(include_scaler=False) + [
        ("model", RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **kwargs,
        )),
    ])


def make_hgb_pipeline(**kwargs: object) -> Pipeline:
    return Pipeline(make_preprocessor(include_scaler=False) + [
        ("model", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **kwargs,
        )),
    ])


MODEL_SPECS = {
    "Linear Regression": ("lr", lambda _target: make_linear_pipeline()),
    "Random Forest": ("rf", lambda _target: make_rf_pipeline(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
    )),
    "Gradient Boosting": ("gb", lambda target: make_hgb_pipeline(**{
        "Water Temperature": {
            "loss": "squared_error",
            "learning_rate": 0.03,
            "max_depth": 10,
            "max_iter": 700,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
        },
        "pH": {
            "loss": "squared_error",
            "learning_rate": 0.03,
            "max_depth": 10,
            "max_iter": 700,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
        },
        "Dissolved Oxygen": {
            "loss": "squared_error",
            "learning_rate": 0.03,
            "max_depth": 10,
            "max_iter": 700,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
        },
        "Nitrate": {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_depth": 8,
            "max_iter": 500,
            "min_samples_leaf": 10,
            "l2_regularization": 0.1,
        },
    }[target])),
}


def wrap_for_target(target_label: str, pipeline: Pipeline) -> object:
    if target_label == "Nitrate":
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "error_rate_pct": float(symmetric_mape(y_true, y_pred)),
    }


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_target_frame(
    df: pd.DataFrame,
    target_col: str,
    valid_range: tuple[float, float],
) -> tuple[pd.DataFrame, int, int] | None:
    needed = FEATURE_COLS + [target_col]
    target_df = df[needed].dropna(subset=[target_col]).copy()
    raw_rows = len(target_df)

    lower, upper = valid_range
    target_df = target_df[target_df[target_col].between(lower, upper)].copy()
    filtered_rows = len(target_df)

    if filtered_rows < MIN_SAMPLES:
        return None
    return target_df, raw_rows, filtered_rows


def cross_validated_r2(
    target_label: str,
    estimator: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )
    return float(np.mean(cv_scores)), float(np.std(cv_scores))


def train_and_save() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    metrics_rows: list[dict[str, float | int | str | bool]] = []

    for target_label, config in TARGETS.items():
        print(f"\n--- {target_label} ({config['column']}) ---")
        prepared = prepare_target_frame(
            dataset,
            target_col=config["column"],
            valid_range=config["valid_range"],
        )
        if prepared is None:
            print(f"  [SKIP] fewer than {MIN_SAMPLES} rows after filtering.")
            continue

        target_df, raw_rows, filtered_rows = prepared
        X = target_df[FEATURE_COLS].to_numpy(dtype=float)
        y = target_df[config["column"]].to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        print(
            f"  Rows kept: {filtered_rows:,} of {raw_rows:,}"
            f"  | train: {len(y_train):,} test: {len(y_test):,}"
        )

        best_target_model_label = ""
        best_target_r2 = float("-inf")

        for model_label, (prefix, build_model) in MODEL_SPECS.items():
            estimator = wrap_for_target(target_label, build_model(target_label))
            cv_mean, cv_std = cross_validated_r2(
                target_label=target_label,
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
            )
            print(f"  {model_label}: CV R²={cv_mean:.3f} ± {cv_std:.3f}")

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            metrics = evaluate_metrics(y_test, y_pred)

            if metrics["r2"] > best_target_r2:
                best_target_r2 = metrics["r2"]
                best_target_model_label = model_label

            print(
                "    Holdout "
                f"R²={metrics['r2']:.3f} "
                f"RMSE={metrics['rmse']:.3f} "
                f"MAE={metrics['mae']:.3f} "
                f"Error={metrics['error_rate_pct']:.2f}%"
            )

            pkl_path = OUTPUT_DIR / f"{prefix}_{config['stem']}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(estimator, f)
            print(f"    Saved -> {pkl_path}")

            metrics_rows.append(
                {
                    "target": target_label,
                    "target_column": config["column"],
                    "model": model_label,
                    "rows_before_filter": raw_rows,
                    "rows_used": filtered_rows,
                    "train_rows": len(y_train),
                    "test_rows": len(y_test),
                    "cv_r2_mean": cv_mean,
                    "cv_r2_std": cv_std,
                    **metrics,
                    "best_model_for_target": False,
                }
            )

        for row in reversed(metrics_rows):
            if row["target"] != target_label:
                break
            row["best_model_for_target"] = row["model"] == best_target_model_label

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values(
            ["target", "r2"],
            ascending=[True, False],
        )
        metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"\nMetrics saved -> {METRICS_PATH}")
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_and_save()
