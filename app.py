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
                              lr_ph.pkl
                              rf_ph.pkl
                              lr_dissolved_oxygen.pkl
                              rf_dissolved_oxygen.pkl
                              lr_nitrate.pkl
                              rf_nitrate.pkl

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
    "Linear Regression": "lr",
    "Random Forest":     "rf",
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
    "Linear Regression": "Fast baseline model with simpler, more linear behavior.",
    "Random Forest": "Ensemble model that captures more complex, non-linear patterns across stations.",
}


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


MODELS   = load_models()
STATIONS = load_station_data()


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
ACCENT       = "#1a6eb5"
ACCENT_LIGHT = "#e8f1fb"
BORDER       = "#d4d4d4"
TEXT_DARK    = "#1a1a1a"
TEXT_MID     = "#555555"
TEXT_LIGHT   = "#888888"
BG_WHITE     = "#ffffff"
BG_PAGE      = "#f5f7fa"
SUCCESS      = "#1e7e34"
DANGER       = "#c0392b"
FONT         = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

CARD_STYLE = {
    "background": BG_WHITE,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER}",
    "padding": "24px",
    "marginBottom": "20px",
}

LABEL_STYLE = {
    "fontFamily": FONT,
    "fontSize": "13px",
    "fontWeight": "600",
    "color": TEXT_MID,
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "marginBottom": "8px",
    "display": "block",
}


# ─────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────
def empty_map_figure() -> go.Figure:
    """Base map with station dots — shown before first prediction."""
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=STATIONS[LAT_COL],
        lon=STATIONS[LON_COL],
        mode="markers",
        marker=dict(size=5, color=ACCENT, opacity=0.45, line=dict(width=0)),
        name="Monitoring stations",
        hovertemplate="<b>Station</b><br>Lat: %{lat:.3f}  Lon: %{lon:.3f}<extra></extra>",
    ))
    _apply_geo_layout(fig)
    return fig


def _apply_geo_layout(fig: go.Figure, height: int = 520) -> None:
    """Apply consistent geo + paper layout to a figure in-place."""
    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,  landcolor="#eef1f5",
            showlakes=True, lakecolor="#d0e8f5",
            showrivers=True, rivercolor="#c0ddf0",
            showcoastlines=True, coastlinecolor=BORDER,
            showsubunits=True,   subunitcolor=BORDER,
            bgcolor=BG_WHITE,
            # Zoomed in to Iowa/Midwest where the stations actually live
            center=dict(lat=42.0, lon=-93.5),
            lataxis_range=[39.0, 45.0],
            lonaxis_range=[-97.5, -89.5],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_WHITE,
        plot_bgcolor=BG_WHITE,
        font=dict(family=FONT),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=12),
        ),
        height=height,
    )


def _stat_tile(label: str, value: str) -> html.Div:
    """Small metric tile for the summary stats panel."""
    return html.Div(
        style={
            "background": BG_PAGE, "borderRadius": "8px",
            "padding": "10px 12px", "textAlign": "center",
        },
        children=[
            html.Div(label, style={
                "fontSize": "11px", "color": TEXT_LIGHT, "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": "4px",
            }),
            html.Div(value, style={
                "fontSize": "15px", "fontWeight": "700", "color": TEXT_DARK,
            }),
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
        label = mt
        if mt == "Random Forest":
            label = "Random Forest (recommended for richer patterns)"
        options.append({
            "label": label if (not target or available) else f"{mt} (unavailable)",
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

_n_loaded           = sum(len(v) for v in MODELS.values())
_model_status_color = SUCCESS if _n_loaded > 0 else DANGER
_model_status_text  = (
    f"{_n_loaded} model(s) loaded" if _n_loaded > 0
    else "No models found — check MODEL_DIR path in config"
)

app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": BG_PAGE, "minHeight": "100vh"},
    children=[

        # ── Header ─────────────────────────────────────
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {ACCENT} 0%, #0d4f8c 100%)",
                "padding": "28px 40px",
                "marginBottom": "28px",
            },
            children=[html.Div(
                style={"maxWidth": "1200px", "margin": "0 auto"},
                children=[
                    html.H1("💧 Water Quality Predictor", style={
                        "color": "white", "margin": "0 0 6px 0",
                        "fontSize": "28px", "fontWeight": "700",
                    }),
                    html.P(
                        "Iowa EPA monitoring stations  ·  "
                        "Pre-trained ML models  ·  "
                        "Spatial interpolation across station network",
                        style={
                            "color": "rgba(255,255,255,0.82)",
                            "margin": 0, "fontSize": "15px",
                        },
                    ),
                ],
            )],
        ),

        # ── Main content ────────────────────────────────
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 24px 40px"},
            children=[

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "340px 1fr",
                        "gap": "20px",
                        "alignItems": "start",
                    },
                    children=[

                        # ── LEFT: Controls ───────────────
                        html.Div([

                            # Model status badge (shows at a glance whether pkls loaded)
                            html.Div(
                                style={
                                    "background": "#f0faf3" if _n_loaded > 0 else "#fdf0ef",
                                    "border": f"1px solid {_model_status_color}",
                                    "borderRadius": "8px",
                                    "padding": "10px 14px",
                                    "fontSize": "13px",
                                    "color": _model_status_color,
                                    "fontWeight": "500",
                                    "marginBottom": "16px",
                                },
                                children=f"{'✅' if _n_loaded > 0 else '❌'}  {_model_status_text}",
                            ),

                            # Target variable
                            html.Div(style=CARD_STYLE, children=[
                                html.Span("Target Variable", style=LABEL_STYLE),
                                dcc.RadioItems(
                                    id="target-radio",
                                    options=_available_targets(),
                                    value=None,
                                    labelStyle={
                                        "display": "flex", "alignItems": "center",
                                        "gap": "10px", "marginBottom": "12px",
                                        "fontSize": "15px", "color": TEXT_DARK,
                                        "cursor": "pointer",
                                    },
                                    inputStyle={
                                        "accentColor": ACCENT,
                                        "width": "16px", "height": "16px",
                                    },
                                ),
                            ]),

                            # Model type
                            html.Div(style=CARD_STYLE, children=[
                                html.Span("Prediction Model", style=LABEL_STYLE),
                                dcc.RadioItems(
                                    id="model-radio",
                                    options=_available_model_types(None),
                                    value="Linear Regression",
                                    labelStyle={
                                        "display": "flex", "alignItems": "center",
                                        "gap": "10px", "marginBottom": "12px",
                                        "fontSize": "15px", "color": TEXT_DARK,
                                        "cursor": "pointer",
                                    },
                                    inputStyle={
                                        "accentColor": ACCENT,
                                        "width": "16px", "height": "16px",
                                    },
                                ),
                                html.Div(
                                    id="model-helper",
                                    style={
                                        "marginTop": "8px",
                                        "padding": "10px 12px",
                                        "background": ACCENT_LIGHT,
                                        "border": f"1px solid {BORDER}",
                                        "borderRadius": "8px",
                                        "fontSize": "13px",
                                        "lineHeight": "1.5",
                                        "color": TEXT_MID,
                                    },
                                ),
                            ]),

                            # Date input + quick-fill buttons
                            html.Div(style=CARD_STYLE, children=[
                                html.Span("Prediction Date", style=LABEL_STYLE),
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=date(2000, 1, 1),
                                    max_date_allowed=date(2030, 12, 31),
                                    placeholder="Select a date…",
                                    display_format="MMM D, YYYY",
                                    style={"width": "100%", "marginBottom": "14px"},
                                ),
                                html.Span("Quick select:", style={
                                    "fontSize": "12px", "color": TEXT_LIGHT,
                                    "display": "block", "marginBottom": "10px",
                                }),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1fr 1fr",
                                        "gap": "8px",
                                    },
                                    children=[
                                        html.Button(
                                            label,
                                            id=f"btn-{label.lower().replace(' ', '-')}",
                                            n_clicks=0,
                                            style={
                                                "background": ACCENT_LIGHT,
                                                "color": ACCENT,
                                                "border": f"1px solid {ACCENT}",
                                                "borderRadius": "8px",
                                                "padding": "7px 6px",
                                                "fontSize": "12px",
                                                "fontWeight": "600",
                                                "cursor": "pointer",
                                                "fontFamily": FONT,
                                            },
                                        )
                                        for label, _ in DATE_SHORTCUTS
                                    ],
                                ),
                            ]),

                            html.Button(
                                "▶  Run Prediction",
                                id="predict-btn",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "background": ACCENT,
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "padding": "14px",
                                    "fontSize": "16px",
                                    "fontWeight": "700",
                                    "cursor": "pointer",
                                    "fontFamily": FONT,
                                    "marginBottom": "12px",
                                    "letterSpacing": "0.02em",
                                },
                            ),

                            html.Div(id="status-msg", style={"marginBottom": "12px"}),
                            html.Div(id="stats-panel"),

                        ]),   # end left panel

                        # ── RIGHT: Map ───────────────────
                        html.Div([
                            html.Div(
                                style={**CARD_STYLE, "padding": "16px"},
                                children=[
                                    html.Div(
                                        id="map-title",
                                        style={
                                            "fontSize": "16px", "fontWeight": "600",
                                            "color": TEXT_DARK,
                                            "marginBottom": "12px",
                                            "padding": "0 8px",
                                        },
                                        children=(
                                            f"{len(STATIONS)} monitoring stations — "
                                            "select options and run a prediction"
                                        ),
                                    ),
                                    dcc.Graph(
                                        id="usa-map",
                                        figure=empty_map_figure(),
                                        config={
                                            "displayModeBar": True,
                                            "modeBarButtonsToRemove": [
                                                "select2d", "lasso2d",
                                            ],
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                            ),
                        ]),   # end right panel

                    ],
                ),

                # ── Footer ──────────────────────────────
                html.Div(
                    style={
                        "textAlign": "center", "padding": "20px 0 0",
                        "fontSize": "13px", "color": TEXT_LIGHT,
                        "borderTop": f"1px solid {BORDER}",
                    },
                    children=(
                        f"{len(STATIONS):,} unique stations  ·  "
                        f"Features: {', '.join(FEATURE_COLS)}  ·  "
                        "Interpolation: cubic spline"
                    ),
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
        return "Choose a model to see what kind of prediction behavior it emphasizes."

    badge_text = "Best for detail" if model_type == "Random Forest" else "Best for interpretability"
    badge_color = ACCENT if model_type == "Random Forest" else "#6c757d"
    return [
        html.Div(
            badge_text,
            style={
                "display": "inline-block",
                "marginBottom": "6px",
                "padding": "3px 8px",
                "borderRadius": "999px",
                "background": badge_color,
                "color": "white",
                "fontSize": "11px",
                "fontWeight": "700",
                "letterSpacing": "0.03em",
                "textTransform": "uppercase",
            },
        ),
        html.Div(MODEL_DESCRIPTIONS.get(model_type, "")),
    ]


# ─────────────────────────────────────────────────────────────
# CALLBACK: Run prediction & update map
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("usa-map",     "figure"),
    Output("status-msg",  "children"),
    Output("map-title",   "children"),
    Output("stats-panel", "children"),
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
            f"⚠️  Please provide {' and '.join(errors)} before running a prediction.",
            style={
                "background": "#fdf0ef",
                "border": f"1px solid {DANGER}",
                "borderRadius": "8px",
                "padding": "12px 14px",
                "fontSize": "14px",
                "color": DANGER,
                "fontWeight": "500",
            },
        )
        return empty_map_figure(), msg, no_update, no_update

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
            f"<b>{target}</b>: %{{marker.color:.2f}} {unit}<br>"
            "Lat: %{lat:.3f}  Lon: %{lon:.3f}<extra></extra>"
        ),
    ))

    # Actual station markers on top
    fig.add_trace(go.Scattergeo(
        lat=station_df[LAT_COL],
        lon=station_df[LON_COL],
        mode="markers",
        marker=dict(
            size=8,
            color=station_df["predicted"],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            line=dict(color="white", width=0.8),
        ),
        name="Monitoring stations",
        hovertemplate=(
            f"<b>Station — {target}</b><br>"
            f"Predicted: %{{marker.color:.2f}} {unit}<br>"
            "Lat: %{lat:.3f}  Lon: %{lon:.3f}<extra></extra>"
        ),
    ))

    _apply_geo_layout(fig)

    # ── Status banner ─────────────────────────────────
    status = html.Div(
        (
            f"✅  {model_type}"
            f"{'  ·  richer pattern mode' if model_type == 'Random Forest' else ''}"
            f"  ·  {target}  ·  {pred_date.strftime('%b %d, %Y')}"
        ),
        style={
            "background": "#f0faf3",
            "border": f"1px solid {SUCCESS}",
            "borderRadius": "8px",
            "padding": "12px 14px",
            "fontSize": "14px",
            "color": SUCCESS,
            "fontWeight": "500",
        },
    )

    title = (
        f"{target} ({unit})  ·  {model_type}  ·  "
        f"Predicted for {pred_date.strftime('%B %d, %Y')}"
    )
    if model_type == "Random Forest":
        title += "  ·  enhanced pattern view"

    # ── Summary stats ─────────────────────────────────
    preds = station_df["predicted"]
    stats = html.Div(
        style={**CARD_STYLE, "padding": "18px"},
        children=[
            html.Span("Prediction Summary", style=LABEL_STYLE),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                children=[
                    _stat_tile("Min",     f"{preds.min():.2f} {unit}"),
                    _stat_tile("Max",     f"{preds.max():.2f} {unit}"),
                    _stat_tile("Mean",    f"{preds.mean():.2f} {unit}"),
                    _stat_tile("Std Dev", f"{preds.std():.2f} {unit}"),
                ],
            ),
        ],
    )

    return fig, status, title, stats


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
