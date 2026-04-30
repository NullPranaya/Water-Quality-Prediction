# Water Quality Prediction

An end-to-end machine learning project for predicting water quality conditions across Iowa EPA monitoring stations. The project combines real water quality measurements, climate records, and agricultural data into a unified modeling pipeline, then serves predictions through an interactive Dash dashboard with map-based visualization.

This repository is maintained as a completed project snapshot, with pre-trained artifacts and processed datasets included for reproducibility.
It is set up so the project can be inspected or run without rebuilding the entire pipeline first.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the App](#how-to-run-the-app)
- [Dashboard Features](#dashboard-features)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [How to Retrain the Models](#how-to-retrain-the-models)
- [Additional Modeling Outputs](#additional-modeling-outputs)
- [Tech Stack](#tech-stack)

---

## Project Overview

This project answers a practical question: given climate conditions on any date, what water quality can we expect at monitoring stations across Iowa?

The pipeline pulls together three real-world datasets — EPA water quality measurements, ISU climate station records, and USDA agricultural data — cleans and merges them, and trains scikit-learn regression models for four water quality targets:

| Target Variable | Unit |
|---|---|
| Water Temperature | °C |
| pH | pH |
| Dissolved Oxygen | mg/L |
| Nitrate | mg/L |

Predictions are served through a locally-runnable Dash app. The user picks a target, a model type, and a date — the app runs inference across all monitoring stations and renders a spatially interpolated map of predicted values across Iowa.

Pre-trained model files are included in the repository so the dashboard works immediately without retraining.

---

## Data Sources

| Dataset | Source | Raw File |
|---|---|---|
| EPA Water Quality Measurements | U.S. Environmental Protection Agency (WQX) | `data/tabular/water-quality/raw/epa-wq.csv` |
| EPA Monitoring Station Locations | U.S. Environmental Protection Agency | `data/tabular/water-quality/raw/epa-stations.csv` |
| Iowa DNR Water Quality | Iowa Department of Natural Resources | `data/tabular/water-quality/raw/IowaDNR-wq.csv` |
| Climate Records | Iowa State University Climate Science | `data/tabular/climate/raw/isu-climate.csv` |
| Agricultural Data | USDA National Agricultural Statistics Service | `data/tabular/agricultural/raw/usdaNass-agriculture.csv` |

Each EPA monitoring station is spatially matched to its nearest ISU climate station, and the climate features for that station are joined to each water quality observation by date.

---

## Repository Structure

```
.
├── app.py                          # Dash dashboard (main entry point)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── tabular/
│   │   ├── water-quality/
│   │   │   ├── raw/                # Original EPA and DNR downloads
│   │   │   └── clean/              # Cleaned and merged EPA tables
│   │   ├── climate/
│   │   │   ├── raw/                # Original ISU climate download
│   │   │   └── clean/              # Cleaned climate table
│   │   ├── agricultural/
│   │   │   ├── raw/                # Original USDA NASS download
│   │   │   └── clean/              # Cleaned agricultural table
│   │   ├── merged/                 # Final joined tables used for modeling
│   │   │   ├── epa-climate-merged.csv        # Main modeling dataset
│   │   │   ├── epa-merged.csv                # EPA stations + measurements joined
│   │   │   └── epa-to-climate-station-map.csv # Nearest-station spatial map
│   │   └── modeling/               # Model evaluation outputs
│   │       ├── multiple_linear_regression_metrics.csv
│   │       └── multiple_linear_regression_coefficients.csv
│   ├── images/
│   │   └── water-images/           # Water quality image samples (clean/dirty)
│   └── text/
│       └── raw/                    # City-level water summary text files
│
└── src/
    ├── cleaning/
    │   └── tabular/
    │       ├── water-quality/      # Notebooks: epa-wq-clean, epa-stations-clean
    │       ├── climate/            # Notebook: climate-clean
    │       └── agricultural/      # Notebook: usdaNass-agriculture-clean
    ├── merge/
    │   ├── merge_epa_climate.py              # Script: join EPA + climate by date/station
    │   ├── merge_epa_climate_ag.py           # Script: add agricultural features
    │   └── *.ipynb                           # Exploratory merge notebooks
    └── modeling/
        ├── train_sklearn_models.py           # Train and save all .pkl models
        ├── multiple_linear_regression.py     # Standalone MLR script (numpy only)
        ├── lr_water_temperature.pkl          # Pre-trained Linear Regression
        ├── rf_water_temperature.pkl          # Pre-trained Random Forest
        ├── lr_ph.pkl
        ├── rf_ph.pkl
        ├── lr_dissolved_oxygen.pkl
        ├── rf_dissolved_oxygen.pkl
        ├── lr_nitrate.pkl
        └── rf_nitrate.pkl
```

---

## Setup and Installation

**Requirements:** Python 3.9 or higher.

### 1. Clone the repository

```bash
git clone https://github.com/NullPranaya/Water-Quality-Prediction.git
cd Water-Quality-Prediction
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything needed: Dash, Plotly, pandas, NumPy, SciPy, scikit-learn, and the full Jupyter environment.

---

## How to Run the App

With your virtual environment activated and dependencies installed, run from the project root:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:8050
```

The app loads all twelve pre-trained `.pkl` models from `src/modeling/` at startup — no retraining needed. A green status badge in the sidebar confirms how many models loaded successfully.

---

## Dashboard Features

The dashboard is organized into a left control panel and a right map panel.

**Left Panel — Controls:**
- **Target Variable** — choose one of four water quality parameters to predict (Water Temperature, pH, Dissolved Oxygen, Nitrate)
- **Prediction Model** — switch between Linear Regression, Random Forest, and Gradient Boosting
- **Prediction Date** — pick any date using the date picker, or use the quick-select buttons (Today, In 1 week, In 2 weeks, In 1 month, In 6 months, In 1 year)
- **Run Prediction** — triggers inference across all monitoring stations

**Right Panel — Map:**
- Displays all Iowa EPA monitoring stations on a scoped U.S. map
- After running a prediction, shows a color-coded interpolated surface across the station network using cubic spline spatial interpolation
- Station markers are overlaid on top of the interpolated grid with per-station predicted values visible on hover
- Prediction summary panel shows Min, Max, Mean, and Std Dev for the current prediction

**Color scales by target:**
- Water Temperature — Red-Yellow-Blue (diverging)
- pH — Red-Yellow-Green
- Dissolved Oxygen — Blues
- Nitrate — Yellow-Orange-Red

---

## Data Pipeline

The project follows a three-stage pipeline. Pre-processed outputs are already committed to the repository so you can skip to running the app, but the full pipeline can be re-run from scratch.

### Stage 1 — Cleaning

Each raw dataset is cleaned in a Jupyter notebook under `src/cleaning/`:

| Notebook | What it does |
|---|---|
| `water-quality/epa-wq-clean.ipynb` | Filters EPA water quality records, pivots from long to wide format (one row per station-date, one column per parameter), standardizes units |
| `water-quality/epa-stations-clean.ipynb` | Cleans station metadata, extracts lat/lon, deduplicates |
| `climate/climate-clean.ipynb` | Parses ISU climate records, standardizes date format, handles missing values |
| `agricultural/usdaNass-agriculture-clean.ipynb` | Cleans USDA NASS agricultural features |

### Stage 2 — Merging

Cleaned tables are joined in `src/merge/`:

1. **EPA stations + measurements** — inner join on `MonitoringLocationIdentifier` to produce `epa-merged.csv`
2. **Spatial matching** — each EPA station is matched to its nearest ISU climate station by haversine distance, producing `epa-to-climate-station-map.csv`
3. **EPA + Climate join** — `merge_epa_climate.py` joins `epa-merged.csv` with `isu-climate-clean.csv` on `(climate_station, date)`, yielding the main modeling table `epa-climate-merged.csv`
4. **Add agricultural features** — `merge_epa_climate_ag.py` optionally extends the merged table with USDA NASS data

The final modeling dataset `data/tabular/merged/epa-climate-merged.csv` contains one row per station-date observation with all water quality targets and climate features side by side.

### Stage 3 — Modeling

`src/modeling/train_sklearn_models.py` trains three model types (Linear Regression, Random Forest, Gradient Boosting) for each of the four targets and saves them as `.pkl` files:

```
lr_water_temperature.pkl    rf_water_temperature.pkl    gb_water_temperature.pkl
lr_ph.pkl                   rf_ph.pkl                   gb_ph.pkl
lr_dissolved_oxygen.pkl     rf_dissolved_oxygen.pkl     gb_dissolved_oxygen.pkl
lr_nitrate.pkl              rf_nitrate.pkl              gb_nitrate.pkl
```

---

## Models

### Features

All three model types are trained on the same twelve inference features:

| Feature | Description |
|---|---|
| `doy` | Day of year (1–366) — primary seasonal signal |
| `gdd_40_86` | Growing degree days (base 40°F, max 86°F) |
| `high` | Daily high temperature (°F) |
| `highc` | Daily high temperature (°C) |
| `low` | Daily low temperature (°F) |
| `lowc` | Daily low temperature (°C) |
| `precip` | Daily precipitation (inches) |
| `snow` | Daily snowfall (inches) |
| `snowd` | Snow depth on ground (inches) |
| `distance_to_climate_station_km` | Distance from EPA station to nearest climate station |
| `LatitudeMeasure` | EPA monitoring station latitude |
| `LongitudeMeasure` | EPA monitoring station longitude |

### Model Architecture

Each saved model is a scikit-learn `Pipeline` that keeps runtime inference
compatible with the same raw input features while refining them during
training:

1. `SimpleImputer(strategy="median")` — fills missing climate values
2. Derived features — cyclical seasonality (`sin/cos(doy)`), temperature range, and aggregate moisture
3. Optional `StandardScaler` for the linear model
4. Model — `LinearRegression`, tuned `RandomForestRegressor`, or `HistGradientBoostingRegressor`

An 80/20 train/test split with `random_state=42` is used for all targets.

### Linear Regression (numpy implementation)

`src/modeling/multiple_linear_regression.py` also includes a from-scratch multiple linear regression implementation using `numpy.linalg.lstsq` (no scikit-learn). This was used to generate the evaluation CSVs in `data/tabular/modeling/`. It supports command-line arguments for data path, output directory, target selection, and minimum sample thresholds.

---

## How to Retrain the Models

If you want to regenerate the `.pkl` files from the merged dataset:

```bash
# Activate your virtual environment first
# macOS / Linux:  source venv/bin/activate
# Windows:        venv\Scripts\activate

python src/modeling/train_sklearn_models.py
```

This will print training progress and evaluation metrics (R², RMSE, MAE) for each target and model type, then overwrite the `.pkl` files in `src/modeling/`.

To also regenerate the CSV modeling outputs using the numpy-based linear regression:

```bash
python src/modeling/multiple_linear_regression.py
```

Optional arguments:
```
--data-path PATH       Path to the merged CSV (default: data/tabular/merged/epa-climate-merged.csv)
--output-dir DIR       Output directory for metrics/coefficients CSVs
--targets NAME [...]   Specific target CharacteristicName values to model
--top-n N              Number of most frequent targets to model (default: 5)
--min-samples N        Minimum rows required to train a target (default: 100)
--test-size FLOAT      Fraction held out for testing (default: 0.2)
--seed INT             Random seed (default: 42)
```

---

## Additional Modeling Outputs

The repository includes pre-generated evaluation outputs from the numpy linear regression:

- [`data/tabular/modeling/multiple_linear_regression_metrics.csv`](data/tabular/modeling/multiple_linear_regression_metrics.csv) — R², RMSE, MAE, sample counts per target
- [`data/tabular/modeling/multiple_linear_regression_coefficients.csv`](data/tabular/modeling/multiple_linear_regression_coefficients.csv) — intercept and feature coefficients per target

---

## Tech Stack

| Category | Libraries |
|---|---|
| Dashboard | Dash 4.x, Plotly 6.x |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (Linear Regression, Random Forest, HistGradientBoosting, Pipeline, SimpleImputer, StandardScaler) |
| Spatial interpolation | SciPy (`griddata` — cubic spline) |
| Geospatial | GeoPandas, Shapely, pyproj, Folium |
| Notebooks | JupyterLab, IPython |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.9+ |

---

## Status

The project is fully functional. Data is cleaned and merged, all twelve trained models are saved in the repository, and the dashboard can be launched locally with a single command for interactive prediction and visualization across Iowa's water monitoring network.
