# Water Quality Prediction

An end-to-end machine learning project for predicting water quality conditions across Iowa monitoring stations. The repository includes data cleaning and merging pipelines, trained regression models, and a Dash dashboard for interactive prediction and visualization.

## Project Overview

This project combines EPA water quality records with climate data to estimate water conditions at monitoring stations across Iowa. The final application loads pre-trained models and generates station-level predictions for:

- Water temperature
- pH
- Dissolved oxygen
- Nitrate

Predictions are displayed in an interactive Dash app with map-based visualization and spatial interpolation across the monitoring network.

## What the Project Includes

- Cleaned EPA, climate, and agricultural datasets
- Merge scripts and notebooks for building the modeling dataset
- Trained scikit-learn regression models saved as `.pkl` files
- A Dash dashboard for local prediction and exploration
- Modeling outputs for linear regression metrics and coefficients

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── tabular/
│   │   ├── agricultural/
│   │   ├── climate/
│   │   ├── merged/
│   │   ├── modeling/
│   │   └── water-quality/
│   ├── images/
│   └── text/
└── src/
    ├── cleaning/
    ├── merge/
    └── modeling/
```

## Dashboard Features

- Select a target variable to predict
- Compare linear regression and random forest models
- Choose a prediction date
- View predictions across Iowa monitoring stations
- See interpolated map surfaces based on station predictions
- Run the app locally with pre-trained models already included in the repo

## Models

The dashboard uses pre-trained scikit-learn pipelines stored in [`src/modeling`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/src/modeling). The current saved models include:

- `lr_water_temperature.pkl`
- `rf_water_temperature.pkl`
- `lr_ph.pkl`
- `rf_ph.pkl`
- `lr_dissolved_oxygen.pkl`
- `rf_dissolved_oxygen.pkl`
- `lr_nitrate.pkl`
- `rf_nitrate.pkl`

These models are trained on climate-related features such as day-of-year, temperature, precipitation, snow, and distance to climate stations.

## Data Pipeline

The project workflow is organized in three main stages:

1. Clean raw EPA, climate, and agricultural datasets in [`src/cleaning`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/src/cleaning).
2. Merge cleaned datasets into modeling tables using scripts in [`src/merge`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/src/merge).
3. Train and save models from the merged dataset using [`src/modeling/train_sklearn_models.py`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/src/modeling/train_sklearn_models.py).

The main runtime dataset used by the dashboard is:

- [`data/tabular/merged/epa-climate-merged.csv`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/data/tabular/merged/epa-climate-merged.csv)

## How to Run the App

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

## How to Retrain the Models

If you want to regenerate the saved scikit-learn models:

```bash
source venv/bin/activate
python src/modeling/train_sklearn_models.py
```

This rewrites the `.pkl` model artifacts in `src/modeling/`.

## Additional Modeling Outputs

The repository also includes multiple linear regression summary outputs:

- [`data/tabular/modeling/multiple_linear_regression_metrics.csv`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/data/tabular/modeling/multiple_linear_regression_metrics.csv)
- [`data/tabular/modeling/multiple_linear_regression_coefficients.csv`](/Users/pranayasigdel/pi515/Water-Quality-Prediction/data/tabular/modeling/multiple_linear_regression_coefficients.csv)

## Tech Stack

- Python
- Dash and Plotly
- pandas and NumPy
- SciPy
- scikit-learn
- Jupyter notebooks

## Status

The project is fully functional in its current form: data is cleaned and merged, trained models are saved in the repository, and the dashboard can be launched locally for interactive prediction and visualization.
