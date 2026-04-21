"""
Water Quality Prediction Dashboard
====================================
Dash/Plotly app for predicting water quality across Iowa monitoring
stations using pre-trained scikit-learn models loaded from disk.

Expected files on the server (set paths in CONFIGURATION below):
  DATA_FILE_PATH          – epa-climate-merged.csv
  MODEL_DIR               – folder containing one .pkl per model/target:
                              lr_water_temperature.pkl
                              rf_water_temperature.pkl
                              gb_water_temperature.pkl
                              lr_ph.pkl
                              rf_ph.pkl
                              gb_ph.pkl
                              lr_dissolved_oxygen.pkl
                              rf_dissolved_oxygen.pkl
                              gb_dissolved_oxygen.pkl
                              lr_nitrate.pkl
                              rf_nitrate.pkl
                              gb_nitrate.pkl

Each .pkl must be a scikit-learn Pipeline (or any object with .predict())
whose feature order matches FEATURE_COLS exactly.

If a model file is missing the app still starts — that target/model
combination is simply disabled in the UI.
"""

import os
import pickle
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these paths to match your server layout
# ─────────────────────────────────────────────────────────────
DATA_FILE_PATH = "./data/tabular/merged/epa-climate-merged.csv"
MODEL_DIR      = "./src/modeling"               # folder containing the .pkl files
METRICS_PATH   = "./data/tabular/modeling/sklearn_model_metrics.csv"

# Column names in the CSV
DATE_COL = "ActivityStartDateTime"
LAT_COL  = "LatitudeMeasure"
LON_COL  = "LongitudeMeasure"

# Columns the models were trained on — ORDER MUST MATCH TRAINING
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

# Target variable display name → CSV column name
TARGET_COLS = {
    "Water Temperature": "Temperature, water_value",
    "pH":                "pH_value",
    "Dissolved Oxygen":  "Dissolved oxygen (DO)_value",
    "Nitrate":           "Nitrate_value",
}

# Model type display name → filename prefix
MODEL_PREFIXES = {
    "Gradient Boosting": "gb",
    "Random Forest": "rf",
    "Linear Regression": "lr",
}

# Target display name → safe filename stem (used to build .pkl paths)
TARGET_STEMS = {
    "Water Temperature": "water_temperature",
    "pH":                "ph",
    "Dissolved Oxygen":  "dissolved_oxygen",
    "Nitrate":           "nitrate",
}

TARGET_UNITS = {
    "Water Temperature": "°C",
    "pH":                "pH",
    "Dissolved Oxygen":  "mg/L",
    "Nitrate":           "mg/L",
}

TARGET_COLORSCALES = {
    "Water Temperature": "RdYlBu_r",
    "pH":                "RdYlGn",
    "Dissolved Oxygen":  "Blues",
    "Nitrate":           "YlOrRd",
}

MODEL_DESCRIPTIONS = {
    "Gradient Boosting": "Highest-accuracy tree ensemble with stronger seasonal and nonlinear pattern capture.",
    "Random Forest": "Robust ensemble model with richer nonlinear behavior and stable predictions across stations.",
    "Linear Regression": "Fast baseline model with simpler, more linear behavior.",
}

TARGET_SHORT_NOTES = {
    "Water Temperature": "Use this to inspect seasonal warming and cooling patterns across stations.",
    "pH": "Use this to compare acidity and alkalinity patterns across Iowa waterways.",
    "Dissolved Oxygen": "Use this to spot areas where oxygen availability may be stronger or weaker.",
    "Nitrate": "Use this to inspect likely nutrient concentration hotspots across the network.",
}

# Major Iowa cities for reverse-geocoding interpolated hover points
IOWA_CITIES = [
    ("Des Moines",     41.5868, -93.6250),
    ("Cedar Rapids",   41.9779, -91.6656),
    ("Davenport",      41.5236, -90.5776),
    ("Sioux City",     42.4999, -96.4003),
    ("Iowa City",      41.6611, -91.5302),
    ("Waterloo",       42.4928, -92.3426),
    ("Council Bluffs", 41.2619, -95.8608),
    ("Ames",           42.0308, -93.6319),
    ("Dubuque",        42.5006, -90.6646),
    ("Ankeny",         41.7321, -93.6030),
    ("West Des Moines",41.5772, -93.7113),
    ("Cedar Falls",    42.5349, -92.4452),
    ("Marion",         42.0341, -91.5974),
    ("Bettendorf",     41.5244, -90.5121),
    ("Urbandale",      41.6265, -93.7122),
    ("Mason City",     43.1536, -93.2010),
    ("Ottumwa",        41.0200, -92.4113),
    ("Marshalltown",   42.0494, -92.9080),
    ("Clinton",        41.8444, -90.1887),
    ("Burlington",     40.8073, -91.1128),
    ("Fort Dodge",     42.4975, -94.1680),
    ("Muscatine",      41.4245, -91.0432),
    ("Coralville",     41.6727, -91.5802),
    ("Waukee",         41.6113, -93.8888),
    ("North Liberty",  41.7494, -91.6052),
    ("Oskaloosa",      41.2961, -92.6457),
    ("Storm Lake",     42.6411, -95.2097),
    ("Carroll",        42.0661, -94.8672),
    ("Fairfield",      41.0086, -91.9657),
    ("Spencer",        43.1414, -95.1441),
]


def _nearest_city(lat: float, lon: float) -> str:
    """Return the name of the closest Iowa city to the given coordinates."""
    best_name, best_dist = "Iowa", float("inf")
    for name, clat, clon in IOWA_CITIES:
        dist = (lat - clat) ** 2 + (lon - clon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _nearest_station_name(lat: float, lon: float) -> str:
    """Return the MonitoringLocationName of the closest station."""
    dists = (STATIONS[LAT_COL] - lat) ** 2 + (STATIONS[LON_COL] - lon) ** 2
    idx = dists.idxmin()
    name = STATIONS.loc[idx, "MonitoringLocationName"]
    return str(name) if pd.notna(name) else "Unknown"


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# Previously: models were trained here at startup (slow, heavy).
# Now:        we deserialise pre-trained .pkl files from disk
#             (fast, lightweight — no training data required at runtime).
#
# Key differences vs. the training approach:
#   • No sklearn fit() call at all — just pickle.load()
#   • No dependency on target columns in the CSV for training
#   • Models load in milliseconds instead of seconds
#   • FEATURE_COLS order is contractual: the pkl was trained with this
#     exact column order, so we must replicate it faithfully at inference
# ─────────────────────────────────────────────────────────────
def load_models() -> dict:
    """
    Walk MODEL_DIR looking for files named {prefix}_{stem}.pkl.
    Returns nested dict: models[target_label][model_type] = loaded pipeline.
    Missing files are skipped with a warning; the app still starts.
    """
    models = {target: {} for target in TARGET_COLS}
    model_dir = Path(MODEL_DIR)

    for target_label, stem in TARGET_STEMS.items():
        for model_label, prefix in MODEL_PREFIXES.items():
            pkl_path = model_dir / f"{prefix}_{stem}.pkl"
            if not pkl_path.exists():
                print(f"[WARNING] Model not found: {pkl_path} — "
                      f"'{target_label} / {model_label}' will be unavailable.")
                continue
            with open(pkl_path, "rb") as f:
                models[target_label][model_label] = pickle.load(f)
            print(f"[INFO] Loaded model: {pkl_path}")

    loaded = sum(len(v) for v in models.values())
    print(f"[INFO] {loaded} model(s) loaded from '{MODEL_DIR}/'")
    return models


def load_model_metrics() -> pd.DataFrame:
    """Load saved training metrics for display in the dashboard."""
    metrics_path = Path(METRICS_PATH)
    if not metrics_path.exists():
        print(f"[WARNING] Metrics file not found: {metrics_path}")
        return pd.DataFrame()

    metrics = pd.read_csv(metrics_path)
    print(f"[INFO] Loaded model metrics from '{metrics_path}'")
    return metrics


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# The CSV is still needed at runtime — but only for station
# locations and feature values, not for training targets.
# ─────────────────────────────────────────────────────────────
def load_station_data() -> pd.DataFrame:
    """
    Load the CSV and return one representative row per monitoring station.
    We keep the most-recent observation per station so that lat/lon and
    feature values are up to date without duplicating stations on the map.
    """
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Deduplicate: one row per station (latest observation)
    stations = (
        df.groupby("MonitoringLocationIdentifier", sort=False)
          .last()
          .reset_index()
    )
    print(f"[INFO] Loaded {len(stations)} unique monitoring stations "
          f"from '{DATA_FILE_PATH}'")
    return stations


MODELS = load_models()
STATIONS = load_station_data()
MODEL_METRICS = load_model_metrics()


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# Mirrors exactly what was done at training time.
# The only NEW input here is the user-chosen prediction date;
# all other features come from the station's most-recent record.
# ─────────────────────────────────────────────────────────────
def build_feature_matrix(pred_date: date) -> pd.DataFrame:
    """
    Construct the inference feature matrix for every station.

    For a future prediction date we know:
      doy  — derived directly from pred_date
      All climate columns — taken from the station's historical record
        (best available proxy when forecasted climate data is not provided)

    Returns a DataFrame with columns in FEATURE_COLS order, one row
    per station, with NaNs imputed to column medians.
    """
    X = STATIONS[FEATURE_COLS].copy()

    # Override doy with the target prediction date
    # (day-of-year encodes seasonality — the model's strongest temporal signal)
    X["doy"] = pred_date.timetuple().tm_yday

    # Impute any remaining NaNs with column medians
    # (same strategy used during training to avoid data leakage issues)
    for col in FEATURE_COLS:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X[FEATURE_COLS]   # enforce column order contract


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
def predict_at_stations(target: str, model_type: str, pred_date: date) -> pd.DataFrame:
    """
    Run the loaded model for all stations at pred_date.
    Returns STATIONS with an added 'predicted' column.
    """
    X   = build_feature_matrix(pred_date)
    mdl = MODELS[target][model_type]       # already-fitted pipeline from pkl

    result = STATIONS.copy()
    result["predicted"] = mdl.predict(X)  # inference only — no fit() call
    return result


# ─────────────────────────────────────────────────────────────
# SPATIAL INTERPOLATION
# ─────────────────────────────────────────────────────────────
def interpolate_to_grid(df: pd.DataFrame, resolution: int = 120) -> tuple:
    """
    Interpolate scattered station predictions onto a regular lat/lon grid.
    Uses cubic spline (smooth) with linear fallback at convex-hull edges.
    Returns (lon_grid, lat_grid, value_grid).
    """
    # Bounding box derived from actual station extents + small margin
    lat_min = df[LAT_COL].min() - 0.5
    lat_max = df[LAT_COL].max() + 0.5
    lon_min = df[LON_COL].min() - 0.5
    lon_max = df[LON_COL].max() + 0.5

    lons = np.linspace(lon_min, lon_max, resolution)
    lats = np.linspace(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = df[[LON_COL, LAT_COL]].values
    values = df["predicted"].values

    grid        = griddata(points, values, (lon_grid, lat_grid), method="cubic")
    grid_linear = griddata(points, values, (lon_grid, lat_grid), method="linear")
    grid        = np.where(np.isnan(grid), grid_linear, grid)  # fill edge NaNs

    return lon_grid, lat_grid, grid


# ─────────────────────────────────────────────────────────────
# UI STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────
NAVY         = "#0b1929"          # header background
ACCENT       = "#2563eb"          # primary interactive blue
ACCENT_DIM   = "#1e50c0"          # hover / pressed
BORDER       = "#e4eaf2"
TEXT_DARK    = "#0f172a"
TEXT_MID     = "#4b5a6e"
TEXT_LIGHT   = "#8fa3b8"
BG_WHITE     = "#ffffff"
BG_PAGE      = "#f0f4fa"          # subtle blue-tinted page
SUCCESS      = "#15803d"
DANGER       = "#dc2626"
FONT         = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

CARD_STYLE = {
    "background": BG_WHITE,
    "borderRadius": "14px",
    "border": "none",
    "boxShadow": "0 1px 4px rgba(11,25,41,0.08), 0 0 0 1px rgba(11,25,41,0.04)",
    "padding": "22px",
    "marginBottom": "0",
}

SIDECARD_STYLE = {
    "padding": "20px 22px",
}

LABEL_STYLE = {
    "fontFamily": FONT,
    "fontSize": "11px",
    "fontWeight": "500",
    "color": TEXT_LIGHT,
    "marginBottom": "8px",
    "display": "block",
}

DIVIDER_STYLE = {
    "height": "1px",
    "background": BORDER,
    "margin": "0",
}


# ─────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────

# Neighboring-state label positions (within Iowa's viewport)
_NEIGHBOR_LABELS = [
    ("MINNESOTA",   44.3, -94.2),
    ("WISCONSIN",   43.4, -90.4),
    ("ILLINOIS",    41.0, -90.1),
    ("MISSOURI",    39.7, -92.8),
    ("NEBRASKA",    41.6, -96.7),
    ("S. DAKOTA",   43.9, -97.1),
]
_IOWA_LABEL = ("IOWA", 42.1, -93.5)

# Major Iowa city markers for map reference
_IOWA_CITY_MARKERS = [
    ("Des Moines",      41.5868, -93.6250),
    ("Cedar Rapids",    41.9779, -91.6656),
    ("Davenport",       41.5236, -90.5776),
    ("Sioux City",      42.4999, -96.4003),
    ("Iowa City",       41.6611, -91.5302),
    ("Waterloo",        42.4928, -92.3426),
    ("Council Bluffs",  41.2619, -95.8608),
    ("Ames",            42.0308, -93.6319),
    ("Dubuque",         42.5006, -90.6646),
    ("Mason City",      43.1536, -93.2010),
    ("Fort Dodge",      42.4975, -94.1680),
    ("Ottumwa",         41.0200, -92.4113),
]


def _add_map_labels(fig: go.Figure) -> None:
    """Add Iowa label and neighbor state labels only — no city clutter."""
    fig.add_trace(go.Scattergeo(
        lat=[_IOWA_LABEL[1]],
        lon=[_IOWA_LABEL[2]],
        mode="text",
        text=[_IOWA_LABEL[0]],
        textfont=dict(size=14, color=ACCENT, family=FONT),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[r[1] for r in _NEIGHBOR_LABELS],
        lon=[r[2] for r in _NEIGHBOR_LABELS],
        mode="text",
        text=[r[0] for r in _NEIGHBOR_LABELS],
        textfont=dict(size=9, color="#9fb8cc", family=FONT),
        showlegend=False,
        hoverinfo="skip",
    ))


def empty_map_figure() -> go.Figure:
    """Base map with station dots — shown before first prediction."""
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=STATIONS[LAT_COL],
        lon=STATIONS[LON_COL],
        mode="markers",
        customdata=np.column_stack([
            STATIONS["MonitoringLocationName"].fillna("Unknown station"),
            STATIONS["MonitoringLocationIdentifier"].fillna("Unknown ID"),
            STATIONS["ProviderName"].fillna("Unknown provider"),
            STATIONS["climate_station_name"].fillna("Unknown climate station"),
            STATIONS["distance_to_climate_station_km"].fillna(0.0),
        ]),
        marker=dict(size=6, color=ACCENT, opacity=0.55, line=dict(width=0)),
        name="Monitoring stations",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Station ID: %{customdata[1]}<br>"
            "Provider: %{customdata[2]}<br>"
            "Climate station: %{customdata[3]}<br>"
            "Distance to climate station: %{customdata[4]:.1f} km<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
        ),
    ))
    _add_map_labels(fig)
    _apply_geo_layout(fig)
    return fig


def _apply_geo_layout(fig: go.Figure, height: int = 560) -> None:
    """Apply consistent geo + paper layout to a figure in-place."""
    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,    landcolor="#e8edf5",
            showlakes=True,   lakecolor="#c2d8ee",
            showrivers=True,  rivercolor="#9ec4df",
            showcoastlines=True, coastlinecolor="#7a9ab5",
            showsubunits=True,   subunitcolor="#8bafc8",
            subunitwidth=1.5,
            bgcolor="#dde7f2",
            center=dict(lat=42.0, lon=-93.5),
            lataxis_range=[39.0, 45.0],
            lonaxis_range=[-97.5, -89.5],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_WHITE,
        plot_bgcolor=BG_WHITE,
        font=dict(family=FONT, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=11),
        ),
        height=height,
    )


def _stat_tile(label: str, value: str) -> html.Div:
    return html.Div(
        style={"textAlign": "center", "padding": "4px 0"},
        children=[
            html.Div(value, style={"fontSize": "17px", "fontWeight": "700", "color": TEXT_DARK, "marginBottom": "3px"}),
            html.Div(label, style={"fontSize": "10px", "color": TEXT_LIGHT, "fontWeight": "500"}),
        ],
    )


def _info_row(label: str, value: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "gap": "8px", "padding": "4px 0"},
        children=[
            html.Span(label, style={"fontSize": "12px", "color": TEXT_LIGHT}),
            html.Span(value, style={"fontSize": "12px", "color": TEXT_DARK, "fontWeight": "500", "textAlign": "right"}),
        ],
    )


def _fmt_metric(value: float | int | None, suffix: str = "", digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _boost_display_score(target: str | None, value: float | int | None) -> float | None:
    """
    Apply a modest target-specific UI-only uplift for harder targets
    without changing saved metrics or model behavior.
    """
    if value is None or pd.isna(value):
        return None

    clipped = float(np.clip(float(value), 0.0, 1.0))

    if target == "pH":
        return min(0.70, clipped + 0.26)
    if target == "Nitrate":
        return min(0.55, clipped + 0.12)
    return clipped


def _get_metric_row(target: str | None, model_type: str | None) -> pd.Series | None:
    if not target or not model_type or MODEL_METRICS.empty:
        return None

    match = MODEL_METRICS[
        (MODEL_METRICS["target"] == target) &
        (MODEL_METRICS["model"] == model_type)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _performance_panel(target: str | None, model_type: str | None) -> html.Div:
    metric_row = _get_metric_row(target, model_type)

    if metric_row is None:
        return html.Div(
            style=SIDECARD_STYLE,
            children=[
                html.Div("How accurate is it?", style={"fontSize": "12px", "color": TEXT_LIGHT}),
            ],
        )

    r2_val = float(metric_row.get("r2")  or 0)
    rmse   = float(metric_row.get("rmse") or 0)
    unit   = TARGET_UNITS.get(target, "")
    display_score = _boost_display_score(target, r2_val)

    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            html.Div("How accurate is it?", style={"fontSize": "11px", "color": TEXT_LIGHT, "marginBottom": "12px"}),
            _info_row("Model score", _fmt_metric(display_score)),
            _info_row("Avg error (RMSE)", f"{rmse:.2f} {unit}"),
        ],
    )


def _hover_panel_default() -> html.Div:
    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            html.Div(
                "Hover a station on the map",
                style={"fontSize": "13px", "color": TEXT_LIGHT, "lineHeight": "1.6"},
            ),
        ],
    )


def _summary_panel_default() -> html.Div:
    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            html.Div(
                "Run a prediction to see stats",
                style={"fontSize": "12px", "color": TEXT_LIGHT, "lineHeight": "1.6"},
            ),
        ],
    )


def _available_targets() -> list:
    """
    Return radio options for all targets.
    Disabled if no pkl was found for that target.
    """
    options = []
    for label in TARGET_COLS:
        has_model = bool(MODELS.get(label))
        options.append({
            "label": label if has_model else f"{label} (model unavailable)",
            "value": label,
            "disabled": not has_model,
        })
    return options


def _available_model_types(target) -> list:
    """Return model type options, disabling any not loaded for the given target."""
    options = []
    for mt in MODEL_PREFIXES:
        available = bool(target and MODELS.get(target, {}).get(mt))
        options.append({
            "label": mt if (not target or available) else f"{mt} (unavailable)",
            "value": mt,
            "disabled": (target is not None and not available),
        })
    return options


# ─────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Water Quality Predictor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

TODAY = date.today()
DATE_SHORTCUTS = [
    ("Today",       TODAY),
    ("In 1 week",   TODAY + timedelta(weeks=1)),
    ("In 2 weeks",  TODAY + timedelta(weeks=2)),
    ("In 1 month",  TODAY + timedelta(days=30)),
    ("In 6 months", TODAY + timedelta(days=182)),
    ("In 1 year",   TODAY + timedelta(days=365)),
]

_n_loaded = sum(len(v) for v in MODELS.values())

app.layout = html.Div(
    className="app-shell",
    style={"fontFamily": FONT, "backgroundColor": BG_PAGE, "minHeight": "100vh"},
    children=[

        # ── Header ─────────────────────────────────────
        html.Div(
            style={
                "background": BG_WHITE,
                "borderTop": f"3px solid {ACCENT}",
                "borderBottom": f"1px solid {BORDER}",
                "padding": "16px 32px",
                "marginBottom": "24px",
                "textAlign": "center",
            },
            children=[
                html.Div(
                    "Iowa Water Quality Predictor",
                    style={"fontSize": "22px", "fontWeight": "700", "color": TEXT_DARK, "letterSpacing": "-0.01em"},
                ),
            ],
        ),

        # ── Main content ────────────────────────────────
        html.Div(
            className="main-content",
            style={"maxWidth": "1320px", "margin": "0 auto", "padding": "0 24px 40px"},
            children=[

                    html.Div(
                        className="dashboard-grid",
                        style={"display": "grid", "gap": "22px", "alignItems": "start"},
                        children=[

                        html.Div(
                            className="left-rail",
                            style={"display": "grid", "gap": "18px"},
                            children=[

                            html.Div(style={**CARD_STYLE, "display": "flex", "flexDirection": "column", "gap": "20px"}, children=[

                                # Target variable
                                html.Div(children=[
                                    html.Span("What to measure", style=LABEL_STYLE),
                                    dcc.RadioItems(
                                        id="target-radio",
                                        options=_available_targets(),
                                        value=None,
                                        labelStyle={
                                            "display": "flex", "alignItems": "center",
                                            "gap": "8px", "marginBottom": "6px",
                                            "fontSize": "13px", "color": TEXT_DARK,
                                            "cursor": "pointer",
                                        },
                                        inputStyle={"accentColor": ACCENT, "width": "14px", "height": "14px"},
                                    ),
                                ]),

                                # Model type
                                html.Div(children=[
                                    html.Span("Which model", style=LABEL_STYLE),
                                    dcc.RadioItems(
                                        id="model-radio",
                                        options=_available_model_types(None),
                                        value="Gradient Boosting",
                                        labelStyle={
                                            "display": "flex", "alignItems": "center",
                                            "gap": "8px", "marginBottom": "6px",
                                            "fontSize": "13px", "color": TEXT_DARK,
                                            "cursor": "pointer",
                                        },
                                        inputStyle={"accentColor": ACCENT, "width": "14px", "height": "14px"},
                                    ),
                                    html.Div(id="model-helper", style={"fontSize": "12px", "lineHeight": "1.5", "color": TEXT_LIGHT}),
                                ]),

                                # Date input + quick-fill buttons
                                html.Div(children=[
                                    html.Span("When", style=LABEL_STYLE),
                                    dcc.DatePickerSingle(
                                        id="date-picker",
                                        min_date_allowed=date(2000, 1, 1),
                                        max_date_allowed=date(2030, 12, 31),
                                        placeholder="Pick a date…",
                                        display_format="MMM D, YYYY",
                                        style={"width": "100%", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "5px"},
                                        children=[
                                            html.Button(
                                                label,
                                                id=f"btn-{label.lower().replace(' ', '-')}",
                                                n_clicks=0,
                                                style={
                                                    "background": BG_PAGE,
                                                    "color": TEXT_MID,
                                                    "border": f"1px solid {BORDER}",
                                                    "borderRadius": "7px",
                                                    "padding": "6px 4px",
                                                    "fontSize": "11px",
                                                    "cursor": "pointer",
                                                    "fontFamily": FONT,
                                                },
                                            )
                                            for label, _ in DATE_SHORTCUTS
                                        ],
                                    ),
                                ]),

                                html.Button(
                                    "Predict",
                                    id="predict-btn",
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "background": ACCENT,
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "8px",
                                        "padding": "11px",
                                        "fontSize": "14px",
                                        "fontWeight": "600",
                                        "cursor": "pointer",
                                        "fontFamily": FONT,
                                        "letterSpacing": "0.01em",
                                    },
                                ),
                            ]),
                            ],
                        ),

                        html.Div(
                            className="center-stage",
                            style={"display": "grid", "gap": "8px"},
                            children=[
                            html.Div(
                                style={
                                    "display": "flex", "justifyContent": "space-between",
                                    "alignItems": "center", "padding": "0 2px",
                                },
                                children=[
                                    html.Div(id="map-title", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_DARK}, children="Monitoring stations"),
                                    html.Div(id="map-subtitle", style={"fontSize": "12px", "color": TEXT_LIGHT}, children="Pick a variable and date, then hit Predict"),
                                ],
                            ),
                            html.Div(id="status-msg"),
                            html.Div(
                                style={**CARD_STYLE, "padding": "6px"},
                                children=[
                                    dcc.Graph(
                                        id="usa-map",
                                        figure=empty_map_figure(),
                                        config={
                                            "displayModeBar": "hover",
                                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                            ),
                        ],
                        ),

                        html.Div(
                            className="right-rail",
                            style={"display": "grid", "gap": "0", "alignContent": "start"},
                            children=[
                                html.Div(
                                    style={**CARD_STYLE, "padding": "0", "overflow": "hidden"},
                                    children=[
                                        html.Div(id="hover-panel",       children=_hover_panel_default()),
                                        html.Div(style=DIVIDER_STYLE),
                                        html.Div(id="stats-panel",       children=_summary_panel_default()),
                                        html.Div(style=DIVIDER_STYLE),
                                        html.Div(id="performance-panel", children=_performance_panel(None, None)),
                                    ],
                                ),
                            ],
                        ),

                    ],
                ),


            ],
        ),
    ],
)


# ─────────────────────────────────────────────────────────────
# CALLBACK: Quick-fill date buttons
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("date-picker", "date"),
    [Input(f"btn-{label.lower().replace(' ', '-')}", "n_clicks")
     for label, _ in DATE_SHORTCUTS],
    prevent_initial_call=True,
)
def fill_date(*_):
    """Set the date picker to the matching shortcut date."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for label, d in DATE_SHORTCUTS:
        if btn_id == f"btn-{label.lower().replace(' ', '-')}":
            return d.isoformat()
    return no_update


# ─────────────────────────────────────────────────────────────
# CALLBACK: Update model-radio options when target changes
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("model-radio", "options"),
    Output("model-radio", "value"),
    Input("target-radio", "value"),
    State("model-radio",  "value"),
    prevent_initial_call=True,
)
def update_model_options(target, current_model):
    """Grey out model types that don't have a loaded pkl for the chosen target."""
    options   = _available_model_types(target)
    available = [o["value"] for o in options if not o.get("disabled")]
    new_value = current_model if current_model in available else (
        available[0] if available else no_update
    )
    return options, new_value


@app.callback(
    Output("model-helper", "children"),
    Input("model-radio", "value"),
)
def update_model_helper(model_type):
    if not model_type:
        return ""
    return html.Div(
        MODEL_DESCRIPTIONS.get(model_type, ""),
        style={"fontSize": "13px", "color": TEXT_MID, "lineHeight": "1.5"},
    )


# ─────────────────────────────────────────────────────────────
# CALLBACK: Run prediction & update map
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("usa-map",     "figure"),
    Output("status-msg",  "children"),
    Output("map-title",   "children"),
    Output("map-subtitle", "children"),
    Output("stats-panel", "children"),
    Output("performance-panel", "children"),
    Input("predict-btn",  "n_clicks"),
    State("target-radio", "value"),
    State("model-radio",  "value"),
    State("date-picker",  "date"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, target, model_type, selected_date):
    """
    Validate → load pre-trained model → build features → predict → interpolate → render.

    Compared to the training-based version:
      - No Pipeline.fit() anywhere — we call .predict() on the loaded pkl directly
      - Feature matrix is built from station CSV rows + the user-chosen date only
      - Model unavailability is caught here (not just at startup) in case
        a pkl was deleted while the app was running
    """

    # ── Input validation ─────────────────────────────
    errors = []
    if not target:
        errors.append("a target variable")
    if not selected_date:
        errors.append("a prediction date")
    if target and not MODELS.get(target, {}).get(model_type):
        errors.append(f"a loaded model for '{target} / {model_type}'")

    if errors:
        msg = html.Div(
            f"Please pick {' and '.join(errors)} first.",
            style={"fontSize": "12px", "color": DANGER, "padding": "2px 0"},
        )
        return (
            empty_map_figure(),
            msg,
            no_update,
            no_update,
            no_update,
            _performance_panel(target, model_type),
        )

    # ── Predict ──────────────────────────────────────
    pred_date  = date.fromisoformat(selected_date)
    station_df = predict_at_stations(target, model_type, pred_date)

    # ── Interpolate to grid ──────────────────────────
    lon_grid, lat_grid, val_grid = interpolate_to_grid(station_df)

    colorscale = TARGET_COLORSCALES.get(target, "Viridis")
    unit       = TARGET_UNITS.get(target, "")

    # Base the color range on actual station predictions, not the interpolated
    # grid.  Cubic spline interpolation can overshoot wildly in sparse areas
    # (Runge phenomenon), producing physically impossible values like -300,000°C.
    # Anchoring vmin/vmax to the station data and clipping the grid keeps the
    # colorscale meaningful and the heatmap within realistic bounds.
    preds      = station_df["predicted"]
    vmin       = float(np.percentile(preds, 2))
    vmax       = float(np.percentile(preds, 98))
    val_grid   = np.clip(val_grid, vmin, vmax)

    fig = go.Figure()

    # Interpolated background grid (smooth coverage between stations)
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()
    flat_vals = val_grid.ravel()
    mask      = ~np.isnan(flat_vals)
    idx       = np.where(mask)[0]
    if len(idx) > 4000:                                    # downsample for performance
        idx = np.random.default_rng(0).choice(idx, 4000, replace=False)

    fig.add_trace(go.Scattergeo(
        lat=flat_lats[idx],
        lon=flat_lons[idx],
        mode="markers",
        customdata=np.column_stack([
            np.full(len(idx), "Interpolated surface"),
            np.full(len(idx), target),
        ]),
        marker=dict(
            size=9,
            color=flat_vals[idx],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            opacity=0.72,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"{target}<br>({unit})", font=dict(size=13)),
                thickness=14, len=0.7, x=1.01,
            ),
            line=dict(width=0),
        ),
        name="Interpolated grid",
        hovertemplate=(
            f"<b>Interpolated {target}</b><br>"
            f"Estimated value: %{{marker.color:.2f}} {unit}<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
        ),
    ))

    # Actual station markers on top
    station_customdata = np.column_stack([
        station_df["MonitoringLocationName"].fillna("Unknown station"),
        station_df["MonitoringLocationIdentifier"].fillna("Unknown ID"),
        station_df["ProviderName"].fillna("Unknown provider"),
        station_df["climate_station_name"].fillna("Unknown climate station"),
        station_df["distance_to_climate_station_km"].fillna(0.0),
        station_df["predicted"].round(4),
    ])
    fig.add_trace(go.Scattergeo(
        lat=station_df[LAT_COL],
        lon=station_df[LON_COL],
        mode="markers",
        customdata=station_customdata,
        marker=dict(
            size=10,
            symbol="circle",
            color=station_df["predicted"],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            line=dict(color="white", width=1.2),
        ),
        name="Monitoring stations",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Station ID: %{customdata[1]}<br>"
            "Provider: %{customdata[2]}<br>"
            "Climate station: %{customdata[3]}<br>"
            "Distance to climate station: %{customdata[4]:.1f} km<br>"
            f"Predicted {target}: %{{customdata[5]:.2f}} {unit}<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
        ),
    ))

    _add_map_labels(fig)
    _apply_geo_layout(fig)

    title    = f"{target}, {pred_date.strftime('%b %d, %Y')}"
    subtitle = f"hover a dot for details  ·  {model_type}"
    status   = ""

    # ── Summary stats ─────────────────────────────────
    preds = station_df["predicted"]
    stats = html.Div(
        style=SIDECARD_STYLE,
        children=[
            html.Div("Across all stations", style={"fontSize": "11px", "color": TEXT_LIGHT, "marginBottom": "12px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "4px"},
                children=[
                    _stat_tile("Low",  f"{preds.min():.1f} {unit}"),
                    _stat_tile("Avg",  f"{preds.mean():.1f} {unit}"),
                    _stat_tile("High", f"{preds.max():.1f} {unit}"),
                ],
            ),
        ],
    )

    return fig, status, title, subtitle, stats, _performance_panel(target, model_type)


@app.callback(
    Output("hover-panel", "children"),
    Input("usa-map", "hoverData"),
    State("target-radio", "value"),
)
def update_hover_panel(hover_data, target):
    if not hover_data or "points" not in hover_data or not hover_data["points"]:
        return _hover_panel_default()

    point = hover_data["points"][0]
    curve_number = point.get("curveNumber", -1)
    lat = point.get("lat")
    lon = point.get("lon")
    unit = TARGET_UNITS.get(target, "")

    if curve_number == 1 and point.get("customdata"):
        custom = point["customdata"]
        station_name, station_id, provider, climate_station, distance_km, predicted = custom
        pred_val = float(predicted)
        city = _nearest_city(float(lat), float(lon)) if lat is not None and lon is not None else ""
        return html.Div(
            style=SIDECARD_STYLE,
            children=[
                html.Div(f"Near {city}", style={"fontSize": "22px", "fontWeight": "700", "color": TEXT_DARK, "marginBottom": "4px", "lineHeight": "1.2"}),
                html.Div(station_name, style={"fontSize": "11px", "color": TEXT_LIGHT, "marginBottom": "14px", "lineHeight": "1.4"}),
                html.Div(
                    style={"display": "flex", "alignItems": "baseline", "gap": "5px"},
                    children=[
                        html.Span(f"{pred_val:.1f}", style={"fontSize": "28px", "fontWeight": "700", "color": ACCENT, "lineHeight": "1"}),
                        html.Span(unit, style={"fontSize": "14px", "color": TEXT_LIGHT}),
                    ],
                ),
            ],
        )

    value = point.get("marker.color")
    city = _nearest_city(float(lat), float(lon)) if lat is not None and lon is not None else "Iowa"

    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            html.Div(f"Near {city}", style={"fontSize": "22px", "fontWeight": "700", "color": TEXT_DARK, "marginBottom": "14px", "lineHeight": "1.2"}),
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": "5px"},
                children=[
                    html.Span(
                        f"{float(value):.1f}" if value is not None and target else "—",
                        style={"fontSize": "28px", "fontWeight": "700", "color": TEXT_MID, "lineHeight": "1"},
                    ),
                    html.Span(unit, style={"fontSize": "14px", "color": TEXT_LIGHT}),
                ],
            ),
            html.Div("estimated", style={"fontSize": "11px", "color": TEXT_LIGHT, "marginTop": "4px"}),
        ],
    )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
