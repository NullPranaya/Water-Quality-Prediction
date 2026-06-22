# Water Quality Prediction

An end-to-end machine learning project for forecasting water quality conditions at Iowa EPA monitoring stations. It combines water quality measurements, climate records, and agricultural data into a unified modeling pipeline, then serves predictions through an interactive Dash dashboard with map-based visualization.

This repository is maintained as a completed project snapshot, with pre-trained artifacts and processed datasets included for reproducibility.
It is organized so the project can be reviewed or run without rebuilding the full pipeline first.

Small documentation-only updates may still be made over time to keep the repository presentation current.

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

This project enables predictive water quality modeling and analysis across Iowa using a comprehensive, multi-source dataset. The core application answers: **given climate conditions on a specific date, what water quality should we expect at monitoring stations across Iowa?**

The repository integrates nine data sources across tabular, spatial, image, and text modalities:

- **Water Quality**: 971K EPA measurements across 1,667 monitoring stations and 80+ parameters
- **Climate**: 4.3M daily records from ISU weather stations and PRISM gridded data
- **Agriculture**: Crop yields, livestock, pesticide application, and nutrient inputs
- **Regulatory**: 3.4M permit discharge records (NPDES) and water quality assessments (ATTAINS)
- **Hydrology**: 561K daily streamflow records from 704 USGS gauges
- **Soil**: 10.5K soil map units with hydrologic and chemical properties
- **Land Use**: 11-year time series of 30 m resolution cropland classification
- **Demographics**: Population and employment trends by county
- **Imagery & Text**: Training images for water quality classification and city-level water summaries

The core modeling pipeline trains scikit-learn regression models for four water quality targets:

| Target Variable | Unit | Use Cases |
|---|---|---|
| Water Temperature | °C | Thermal ecology, aquatic life habitat |
| pH | pH | Acid mine drainage detection, alkalinity trends |
| Dissolved Oxygen | mg/L | Hypoxia risk, aquatic stress prediction |
| Nitrate | mg/L | Nutrient loading, agricultural runoff impacts |

Predictions are delivered through a locally runnable Dash app. The user selects a target, a model type, and a date, and the app runs inference across all monitoring stations before rendering a spatially interpolated map of predicted values across Iowa.

Pre-trained model files are included in the repository so the dashboard works immediately without retraining. Beyond prediction, the assembled data supports watershed-scale analysis, permit compliance evaluation, and agricultural influence studies.

---

## Data Sources

This project assembles a comprehensive, multi-modal dataset covering water quality, climate, agriculture, regulatory compliance, streamflow, soil, and census data for Iowa.

### Water Quality (557 MB raw + 10 MB clean)
| Dataset | Source | File | Records |
|---|---|---|---|
| EPA Water Quality Measurements | U.S. EPA (WQX) | `data/tabular/water-quality/raw/epa-wq.csv` | 971K observations |
| EPA Monitoring Stations | U.S. EPA | `data/tabular/water-quality/raw/epa-stations.csv` | 1,667 stations |
| Cleaned & Pivoted Data | (processed) | `data/tabular/water-quality/clean/epa-wq-clean.csv` | 79K station-date records |

80+ measured water quality parameters including temperature, pH, dissolved oxygen, nitrate, phosphate, turbidity, conductance, chlorophyll, bacteria (E. coli), and trace metals.

### Climate (254 MB raw + 13 MB clean)
| Dataset | Source | File | Records |
|---|---|---|---|
| ISU Climate Stations | Iowa State University | `data/tabular/climate/raw/isu-climate.csv` | 221K daily records |
| PRISM Climate Grid | Oregon State University | `data/tabular/climate/raw/prism-iowa-climate.csv` | 4.1M daily grid cells |
| Cleaned Climate Data | (processed) | `data/tabular/climate/clean/isu-climate-clean.csv` | 221K records |

Daily measurements: max/min temperature, dew point, precipitation, snowfall, snow depth, wind speed/direction, and humidity.

### Agriculture (7 MB)
| Dataset | Source | File | Size |
|---|---|---|---|
| Crop Yields | USDA NASS | `data/tabular/01_raw/agriculture/USDA-NASS-Crop-Yields.csv` | 1.8 MB |
| Livestock Inventory | USDA NASS | `data/tabular/01_raw/agriculture/USDA-NASS-Livestock-Inventory.csv` | 4.7 MB |
| Crop Chemical Application | USDA NASS | `data/tabular/01_raw/agriculture/USDA-NASS-Crop-Chemical-Application.csv` | 108 KB |
| Fertilizer Spending | USDA NASS | `data/tabular/01_raw/agriculture/USDA-NASS-Chemical-Fertilizer-Feed-Spending.csv` | 23 KB |
| N-P Nutrient Inputs | USDA | `data/tabular/01_raw/agriculture/N-P_from_*.xlsx` | 1950–2017 |

### Conservation BMPs (Iowa NRS Tracking, 3.6 MB)
| Dataset | Source | File | Records |
|---|---|---|---|
| NRS Tracking — Full | Iowa State University / INRS | `data/tabular/01_raw/bmp/iowa-nrs-tracking.csv` | 20,428 rows (2003–2022) |
| NRS Tracking — HUC-8 BMP Practices | Iowa State University / INRS | `data/tabular/01_raw/bmp/iowa-nrs-bmp-huc8.csv` | 2,195 rows |

Annual practice adoption counts and acres by HUC-8 watershed (56 watersheds, 2003–2022) for CREP wetlands, bioreactors, saturated buffers, cover crops (NRCS practice 340), and erosion control structures. Joins to NHDPlus HUC-12 boundaries on the first 8 digits of the HUC-12 code.

### NPDES Compliance & Permits (850 MB, 3.4M records)
| Dataset | Source | File | Records |
|---|---|---|---|
| Discharge Monitoring Reports | EPA NPDES | `data/tabular/01_raw/npdes/NPDES_DMRS_FY*.csv` | 308K–365K per year (2015–2025) |
| Water Quality Assessments | EPA ATTAINS | `data/tabular/01_raw/npdes/NPDES_ATTAINS_AU_SUMMARIES.csv` | 820K assessment units |
| NPDES Catchments | EPA | `data/tabular/01_raw/npdes/NPDES_CATCHMENTS.csv` | Catchment-level data |
| ECHO Facility Metadata | EPA ECHO (ICIS) | `data/tabular/01_raw/npdes/echo-facilities-iowa.csv` | 2,216 Iowa NPDES facilities |
| ECHO NAICS Codes | EPA ECHO | `data/tabular/01_raw/npdes/echo-naics-iowa.csv` | 1,668 permit-NAICS associations |
| ECHO SIC Codes | EPA ECHO | `data/tabular/01_raw/npdes/echo-sics-iowa.csv` | 1,693 permit-SIC associations |

Detailed permit-linked discharge monitoring with effluent limits, reported exceedances, violations, and water body impairment status. ECHO facility data adds treatment-side context: facility type code, geocoded location, and SIC/NAICS industry classification (866 sewage treatment facilities and 168 cattle feedlots identified). All 1,469 DMR permit numbers match an ECHO facility. Join on `npdes_id` ↔ `EXTERNAL_PERMIT_NMBR`.

### Streamflow (24 MB, 561K records)
| Dataset | Source | File | Records |
|---|---|---|---|
| USGS Discharge Data | USGS | `data/tabular/streamflow/raw/usgs-iowa-discharge.csv` | 561K daily measurements |
| USGS Gauge Locations | USGS | `data/tabular/streamflow/raw/usgs-iowa-gauges.csv` | 704 gauges |

Daily streamflow (cubic feet per second) at 704 monitoring gauges across Iowa.

### Soil (1.3 MB, 10.5K map units)
| Dataset | Source | File |
|---|---|---|
| SSURGO Soil Properties | NRCS | `data/tabular/soil/raw/ssurgo-iowa-attributes.csv` |
| SSURGO Soil Polygons | NRCS | `data/spatial/ssurgo/iowa-mapunit-polygons.shp` |

Soil characteristics: hydraulic group, drainage class, saturated hydraulic conductivity (Ksat), and available water capacity.

### Land Use & Spatial (1.4 MB tabular + rasters)
| Dataset | Source | File |
|---|---|---|
| Cropland Data Layer (CDL) Fractions | NASS | `data/tabular/landuse/cdl-huc12-fractions.csv` |
| CDL Raster Layers | NASS | `data/spatial/cdl/cdl_iowa_YYYY.tif` (2015–2025) |
| NHDPlus Watersheds (HUC-12) | USGS | `data/spatial/nhdplus/wbd-huc12-iowa/WBDSnapshot_Iowa.shp` |
| NHDPlus Gage Locations | USGS | `data/spatial/nhdplus/gage-loc/` |

30 m resolution cropland classification and watershed boundaries.

### Census (demographic)
| Dataset | Source | File |
|---|---|---|
| Census 2010–2019 | U.S. Census | `data/tabular/census/raw/Iowa_Census-2010-2019.xlsx` |
| Census 2020–2025 | U.S. Census | `data/tabular/census/raw/Iowa-Census-2020-2025.xlsx` |

Population, employment, and demographic trends by county.

### Image Data (40 samples)
| Dataset | Source | Samples |
|---|---|---|
| Water Quality Images | (training set) | 24 clean + 16 dirty water samples |

Training dataset for water quality visual classification.

### Text Data (20 summaries)
| Dataset | Source | Files |
|---|---|---|
| City-level Water Summaries | (processed narratives) | Ankeny, Cedar Rapids, Council Bluffs, Des Moines, Davenport, and 15 other Iowa cities |

---

### Data Integration & Linkage

- **Spatial matching**: Each EPA monitoring station is linked to its nearest ISU climate station via haversine distance
- **Temporal alignment**: All sources indexed by date (day, month, year) for time-series analysis
- **Watershed integration**: HUC-12 codes link water quality observations, NPDES permits, and land use
- **County linkage**: Agricultural and census data aggregated by county and joined spatially to monitoring locations

**Geographic coverage**: Iowa statewide (1,667 EPA stations, 704 USGS gauges, 10.5K soil map units)  
**Temporal coverage**: Climate and streamflow span 20+ years; NPDES monitoring FY2015–FY2025; agricultural data from 1950 onward

---

## Repository Structure

```
.
├── app.py                          # Dash dashboard (main entry point)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── tabular/
│   │   ├── water-quality/          # EPA & DNR measurements + station metadata
│   │   │   ├── raw/                # 971K EPA WQX records, 1,667 station locations
│   │   │   └── clean/              # Pivoted 79K station-date records
│   │   ├── climate/                # ISU & PRISM climate data
│   │   │   ├── raw/                # ISU (221K records) + PRISM (4.1M grid cells)
│   │   │   └── clean/              # Standardized daily climate records
│   │   ├── agriculture/            # USDA NASS crop & livestock data
│   │   │   ├── raw/                # Yields, chemical application, livestock, fertilizer
│   │   │   └── clean/              # Processed agricultural features
│   │   ├── streamflow/             # USGS discharge measurements & gauge locations
│   │   │   ├── raw/                # 561K daily streamflow records, 704 gauges
│   │   │   └── clean/              # Processed streamflow data
│   │   ├── npdes/                  # EPA permit compliance & water quality assessments
│   │   │   └── raw/                # DMRS FY2015–2025, ATTAINS, catchments
│   │   ├── soil/                   # NRCS SSURGO soil properties
│   │   │   └── raw/                # 10.5K soil map units with attributes
│   │   ├── landuse/                # NASS CDL fractions by watershed
│   │   ├── census/                 # U.S. Census demographic data
│   │   │   └── raw/                # 2010–2025 population & employment
│   │   ├── merged/                 # Final joined tables used for modeling
│   │   │   ├── epa-climate-merged.csv        # Main modeling dataset
│   │   │   ├── epa-merged.csv                # EPA stations + measurements joined
│   │   │   └── epa-to-climate-station-map.csv # Nearest-station spatial map
│   │   └── modeling/               # Model evaluation outputs
│   │       ├── sklearn_model_metrics.csv
│   │       ├── multiple_linear_regression_metrics.csv
│   │       └── multiple_linear_regression_coefficients.csv
│   ├── spatial/
│   │   ├── cdl/                    # NASS Cropland Data Layer (2015–2025 rasters)
│   │   ├── ssurgo/                 # NRCS soil map unit polygons (shapefiles)
│   │   └── nhdplus/                # USGS National Hydrography Dataset
│   │       ├── wbd-huc12-iowa/     # HUC-12 watershed boundaries
│   │       ├── gage-loc/           # Stream gauge locations
│   │       └── crosswalk/          # Hydrologic feature linkages
│   ├── images/
│   │   └── water-images/           # Training images (24 clean + 16 dirty samples)
│   │       └── train/
│   └── text/
│       └── raw/                    # City-level water summary narratives (20 cities)
│
└── src/
    ├── 01_download/
    │   ├── cdl-cropland-download.ipynb       # NASS CDL rasters
    │   ├── prism-climate-download.ipynb      # PRISM gridded climate
    │   ├── ssurgo-soil-download.ipynb        # NRCS SSURGO soil polygons
    │   ├── usgs-streamflow-download.ipynb    # USGS daily discharge
    │   ├── echo-facilities-download.ipynb    # EPA ECHO NPDES facility metadata
    │   └── iowa-nrs-bmp-download.ipynb       # Iowa NRS conservation BMP tracking
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
        ├── gb_water_temperature.pkl          # Pre-trained Gradient Boosting
        ├── lr_ph.pkl
        ├── rf_ph.pkl
        ├── gb_ph.pkl
        ├── lr_dissolved_oxygen.pkl
        ├── rf_dissolved_oxygen.pkl
        ├── gb_dissolved_oxygen.pkl
        ├── lr_nitrate.pkl
        ├── rf_nitrate.pkl
        └── gb_nitrate.pkl
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

With your virtual environment activated and dependencies installed, run the app from the project root:

```bash
python3 app.py
```

Then open your browser and go to:

```
http://127.0.0.1:8050
```

The app loads all twelve pre-trained `.pkl` models from `src/modeling/` at startup — no retraining needed. A green status badge in the sidebar confirms how many models loaded successfully.

To run the lightweight regression tests:

```bash
./venv/bin/python -m unittest test_app.py test_model_feature_engineering.py
```

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

The repository includes pre-generated evaluation outputs for both training paths:

- [`data/tabular/modeling/sklearn_model_metrics.csv`](data/tabular/modeling/sklearn_model_metrics.csv) — held-out metrics for the saved scikit-learn dashboard models
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

The project is fully functional. The data has been cleaned and merged, all twelve trained models are stored in the repository, and the dashboard can be launched locally with a single command for interactive prediction and visualization across Iowa's water monitoring network.
Lightweight regression tests are also included for the dashboard and feature-engineering path.
