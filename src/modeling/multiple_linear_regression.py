from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/tabular/merged/epa-climate-merged.csv")
OUTPUT_DIR = Path("data/tabular/modeling")
PREDICTOR_COLUMNS = [
    "latitude",
    "longitude",
    "distance_to_climate_station_km",
    "doy",
    "gdd_40_86",
    "high",
    "low",
    "precip",
    "snow",
    "snowd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple linear regression models for water quality targets."
    )
    parser.add_argument(
        "--data-path",
        default=str(DATA_PATH),
        help="CSV file containing EPA and climate features.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for metrics and coefficient CSV files.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of CharacteristicName values to model.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of most frequent target variables to model when --targets is omitted.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of complete rows required to fit a target model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    return parser.parse_args()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(x_design, y_train, rcond=None)
    return coefficients


def predict(coefficients: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x_values)), x_values])
    return x_design @ coefficients


def select_targets(
    df: pd.DataFrame,
    explicit_targets: list[str] | None,
    min_samples: int,
    top_n: int,
) -> list[str]:
    print("Available columns:", df.columns.tolist())
    complete_rows = df.dropna(subset=PREDICTOR_COLUMNS + ["ResultMeasureValue"])
    counts = complete_rows["CharacteristicName"].value_counts()

    if explicit_targets:
        selected = []
        for target in explicit_targets:
            if counts.get(target, 0) >= min_samples:
                selected.append(target)
        return selected

    return counts[counts >= min_samples].head(top_n).index.tolist()


def train_for_target(
    df: pd.DataFrame,
    target_name: str,
    min_samples: int,
    test_size: float,
    seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | str]]] | None:
    target_df = (
        df.loc[df["CharacteristicName"] == target_name, PREDICTOR_COLUMNS + ["ResultMeasureValue"]]
        .dropna()
        .copy()
    )

    if len(target_df) < min_samples:
        return None

    x = target_df[PREDICTOR_COLUMNS].to_numpy(dtype=float)
    y = target_df["ResultMeasureValue"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(target_df))
    split_index = max(1, int(len(indices) * (1 - test_size)))

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    coefficients = fit_linear_regression(x_train, y_train)
    predictions = predict(coefficients, x_test)

    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
    mae = float(np.mean(np.abs(y_test - predictions)))

    metric_row = {
        "target_variable": target_name,
        "sample_count": int(len(target_df)),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2_score(y_test, predictions),
    }

    coefficient_rows = [{"target_variable": target_name, "feature": "intercept", "coefficient": float(coefficients[0])}]
    coefficient_rows.extend(
        {
            "target_variable": target_name,
            "feature": feature_name,
            "coefficient": float(coefficient),
        }
        for feature_name, coefficient in zip(PREDICTOR_COLUMNS, coefficients[1:])
    )

    return metric_row, coefficient_rows


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    targets = select_targets(df, args.targets, args.min_samples, args.top_n)

    if not targets:
        raise ValueError("No target variables met the sample threshold.")

    metrics: list[dict[str, float | int | str]] = []
    coefficients: list[dict[str, float | str]] = []

    for target in targets:
        result = train_for_target(
            df=df,
            target_name=target,
            min_samples=args.min_samples,
            test_size=args.test_size,
            seed=args.seed,
        )
        if result is None:
            continue
        metric_row, coefficient_rows = result
        metrics.append(metric_row)
        coefficients.extend(coefficient_rows)

    if not metrics:
        raise ValueError("Targets were selected, but no models could be trained.")

    metrics_df = pd.DataFrame(metrics).sort_values("r2", ascending=False)
    coefficients_df = pd.DataFrame(coefficients)

    metrics_path = output_dir / "multiple_linear_regression_metrics.csv"
    coefficients_path = output_dir / "multiple_linear_regression_coefficients.csv"

    metrics_df.to_csv(metrics_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)

    print("Multiple linear regression training complete.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Coefficients saved to: {coefficients_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
