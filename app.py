"""
Water Quality Prediction Dashboard
====================================
A Dash/Plotly app for predicting water quality variables
across the USA using Multiple Linear Regression or Random Forest.

CSV Requirements (set DATA_FILE_PATH below):
  Required columns : date, lat, lon
  Target variables : temperature, pH, dissolved_oxygen, nitrate
  Feature columns  : elevation, flow_rate (+ any others listed in FEATURE_COLS)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.interpolate import griddata

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these to match your CSV
# ─────────────────────────────────────────────
DATA_FILE_PATH = "water_quality_data.csv"   # path to your CSV file
DATE_COL       = "date"                     # name of the date column
LAT_COL        = "lat"                      # name of the latitude column
LON_COL        = "lon"                      # name of the longitude column

# Columns used as input features for the models
FEATURE_COLS = ["elevation", "flow_rate"]   # extend with any extra columns

# Target variable column names in the CSV
TARGET_COLS = {
    "Water Temperature": "temperature",
    "pH":                "pH",
    "Dissolved Oxygen":  "dissolved_oxygen",
    "Nitrate":           "nitrate",
}

# Units shown on the colour-bar legend
TARGET_UNITS = {
    "Water Temperature": "°C",
    "pH":                "pH",
    "Dissolved Oxygen":  "mg/L",
    "Nitrate":           "mg/L",
}

# Colour scales for each target variable
TARGET_COLORSCALES = {
    "Water Temperature": "RdYlBu_r",
    "pH":                "RdYlGn",
    "Dissolved Oxygen":  "Blues",
    "Nitrate":           "YlOrRd",
}


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# (used only when DATA_FILE_PATH is not found)
# ─────────────────────────────────────────────
def _generate_synthetic_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-ish synthetic water quality data across the USA."""
    rng = np.random.default_rng(seed)

    # Random stations scattered across the contiguous USA
    lats = rng.uniform(25.0, 49.0, n)
    lons = rng.uniform(-125.0, -66.0, n)

    # Date range: last 3 years, random sample
    start = date(2022, 1, 1).toordinal()
    end   = date.today().toordinal()
    dates = [date.fromordinal(int(d)) for d in rng.integers(start, end, n)]

    # Features with mild geographic signal
    elevation  = np.clip(rng.normal(400, 300, n) + (49 - lats) * 30, 0, 4000)
    flow_rate  = np.clip(rng.exponential(50, n), 0.5, 500)

    # Targets with simple seasonal + geographic patterns + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    season      = np.sin(2 * np.pi * day_of_year / 365)

    temperature     = 15 + 10 * season - 0.004 * elevation + rng.normal(0, 2, n)
    pH_vals         = 7.2 + 0.5 * season - 0.0001 * elevation + rng.normal(0, 0.3, n)
    dissolved_oxygen= 9 - 0.3 * temperature + 0.001 * elevation + rng.normal(0, 0.5, n)
    nitrate         = np.clip(2 + 0.01 * flow_rate - 0.001 * elevation + rng.normal(0, 1, n), 0, 20)

    return pd.DataFrame({
        DATE_COL:           dates,
        LAT_COL:            lats,
        LON_COL:            lons,
        "elevation":        elevation,
        "flow_rate":        flow_rate,
        "temperature":      temperature,
        "pH":               pH_vals,
        "dissolved_oxygen": dissolved_oxygen,
        "nitrate":          nitrate,
    })


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load CSV from disk; fall back to synthetic data if file not found."""
    try:
        df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
        df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
        print(f"[INFO] Loaded {len(df)} rows from '{DATA_FILE_PATH}'")
    except FileNotFoundError:
        print(f"[WARNING] '{DATA_FILE_PATH}' not found — using synthetic data.")
        df = _generate_synthetic_data()
    return df


DF = load_data()


# ─────────────────────────────────────────────
# MODEL TRAINING  (one model per target × model type)
# ─────────────────────────────────────────────
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric feature matrix from the dataframe."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()

    # Add time-based features derived from the date column
    if DATE_COL in df.columns:
        dates     = pd.to_datetime(df[DATE_COL])
        X["doy"]  = dates.dt.dayofyear          # day of year (captures seasonality)
        X["year"] = dates.dt.year
    return X.fillna(X.mean(numeric_only=True))


def train_models(df: pd.DataFrame) -> dict:
    """
    Train both model types for all target variables.
    Returns a nested dict: models[target_label][model_type] = fitted pipeline.
    """
    X = _build_features(df)
    models = {}

    for label, col in TARGET_COLS.items():
        if col not in df.columns:
            continue
        y = df[col].fillna(df[col].mean())
        models[label] = {
            "Linear Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LinearRegression()),
            ]).fit(X, y),

            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )),
            ]).fit(X, y),
        }

    print("[INFO] Models trained for:", list(models.keys()))
    return models


MODELS = train_models(DF)


# ─────────────────────────────────────────────
# PREDICTION + INTERPOLATION HELPERS
# ─────────────────────────────────────────────
def predict_at_stations(target: str, model_type: str, pred_date: date) -> pd.DataFrame:
    """
    Run the chosen model for each station in DF at a given future date.
    Returns the station dataframe with a 'predicted' column.
    """
    df = DF.copy()

    # Replace date column with the requested prediction date
    df[DATE_COL] = pred_date
    X = _build_features(df)

    pipeline   = MODELS[target][model_type]
    df["predicted"] = pipeline.predict(X)
    return df


def interpolate_to_grid(
    df: pd.DataFrame,
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered station predictions onto a regular lat/lon grid
    using cubic interpolation (falls back to linear at grid edges).
    Returns lon_grid, lat_grid, value_grid arrays.
    """
    # USA bounding box
    lat_min, lat_max = 24.0, 50.0
    lon_min, lon_max = -125.0, -66.0

    lons = np.linspace(lon_min, lon_max, resolution)
    lats = np.linspace(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = df[[LON_COL, LAT_COL]].values
    values = df["predicted"].values

    # Cubic for smooth gradient; linear fills convex-hull gaps
    grid = griddata(points, values, (lon_grid, lat_grid), method="cubic")
    grid_linear = griddata(points, values, (lon_grid, lat_grid), method="linear")
    grid = np.where(np.isnan(grid), grid_linear, grid)   # fill edge NaNs

    return lon_grid, lat_grid, grid


# ─────────────────────────────────────────────
# COLOUR PALETTE & STYLE CONSTANTS
# ─────────────────────────────────────────────
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

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

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


# ─────────────────────────────────────────────
# EMPTY MAP FIGURE (shown before first prediction)
# ─────────────────────────────────────────────
def empty_map_figure() -> go.Figure:
    """Render the USA base map with station dots but no heatmap."""
    fig = go.Figure()

    # Station scatter markers
    fig.add_trace(go.Scattergeo(
        lat=DF[LAT_COL],
        lon=DF[LON_COL],
        mode="markers",
        marker=dict(size=5, color=ACCENT, opacity=0.45, line=dict(width=0)),
        name="Monitoring stations",
        hovertemplate="<b>Station</b><br>Lat: %{lat:.2f}  Lon: %{lon:.2f}<extra></extra>",
    ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True, landcolor="#eef1f5",
            showlakes=True, lakecolor="#d0e8f5",
            showrivers=True, rivercolor="#c0ddf0",
            showcoastlines=True, coastlinecolor=BORDER,
            showsubunits=True, subunitcolor=BORDER,
            bgcolor=BG_WHITE,
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
        height=520,
    )
    return fig


# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Water Quality Predictor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Quick-fill date options
TODAY = date.today()
DATE_SHORTCUTS = [
    ("Today",       TODAY),
    ("In 1 week",   TODAY + timedelta(weeks=1)),
    ("In 2 weeks",  TODAY + timedelta(weeks=2)),
    ("In 1 month",  TODAY + timedelta(days=30)),
    ("In 6 months", TODAY + timedelta(days=182)),
    ("In 1 year",   TODAY + timedelta(days=365)),
]

app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": BG_PAGE, "minHeight": "100vh"},
    children=[

        # ── Header ──────────────────────────────────────
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {ACCENT} 0%, #0d4f8c 100%)",
                "padding": "28px 40px",
                "marginBottom": "28px",
            },
            children=[
                html.Div(
                    style={"maxWidth": "1200px", "margin": "0 auto"},
                    children=[
                        html.H1(
                            "💧 Water Quality Predictor",
                            style={
                                "color": "white", "margin": "0 0 6px 0",
                                "fontSize": "28px", "fontWeight": "700",
                            },
                        ),
                        html.P(
                            "Predict water quality variables across the USA "
                            "using machine learning models trained on monitoring station data.",
                            style={
                                "color": "rgba(255,255,255,0.82)",
                                "margin": 0, "fontSize": "15px",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # ── Main content wrapper ─────────────────────────
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 24px 40px"},
            children=[

                # ── Two-column layout ───────────────────
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "340px 1fr",
                        "gap": "20px",
                        "alignItems": "start",
                    },
                    children=[

                        # ── LEFT PANEL: Controls ─────────
                        html.Div([

                            # Target variable
                            html.Div(style=CARD_STYLE, children=[
                                html.Span("Target Variable", style=LABEL_STYLE),
                                dcc.RadioItems(
                                    id="target-radio",
                                    options=[
                                        {"label": label, "value": label}
                                        for label in TARGET_COLS
                                    ],
                                    value=None,
                                    labelStyle={
                                        "display": "flex", "alignItems": "center",
                                        "gap": "10px", "marginBottom": "12px",
                                        "fontSize": "15px", "color": TEXT_DARK,
                                        "cursor": "pointer",
                                    },
                                    inputStyle={"accentColor": ACCENT, "width": "16px", "height": "16px"},
                                ),
                            ]),

                            # Model type
                            html.Div(style=CARD_STYLE, children=[
                                html.Span("Prediction Model", style=LABEL_STYLE),
                                dcc.RadioItems(
                                    id="model-radio",
                                    options=[
                                        {"label": "Multiple Linear Regression",
                                         "value": "Linear Regression"},
                                        {"label": "Random Forest",
                                         "value": "Random Forest"},
                                    ],
                                    value="Linear Regression",
                                    labelStyle={
                                        "display": "flex", "alignItems": "center",
                                        "gap": "10px", "marginBottom": "12px",
                                        "fontSize": "15px", "color": TEXT_DARK,
                                        "cursor": "pointer",
                                    },
                                    inputStyle={"accentColor": ACCENT, "width": "16px", "height": "16px"},
                                ),
                            ]),

                            # Date input
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

                            # Predict button + status message
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

                            # Validation / status message
                            html.Div(id="status-msg", style={"marginBottom": "12px"}),

                            # Stats panel (shows after prediction)
                            html.Div(id="stats-panel"),

                        ]),  # end left panel

                        # ── RIGHT PANEL: Map ─────────────
                        html.Div([
                            html.Div(
                                style={**CARD_STYLE, "padding": "16px"},
                                children=[
                                    html.Div(
                                        id="map-title",
                                        style={
                                            "fontSize": "16px", "fontWeight": "600",
                                            "color": TEXT_DARK, "marginBottom": "12px",
                                            "padding": "0 8px",
                                        },
                                        children="Monitoring Stations — Select options and run a prediction",
                                    ),
                                    dcc.Graph(
                                        id="usa-map",
                                        figure=empty_map_figure(),
                                        config={
                                            "displayModeBar": True,
                                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                            ),
                        ]),  # end right panel

                    ],
                ),  # end grid

                # ── Footer ───────────────────────────────
                html.Div(
                    style={
                        "textAlign": "center", "padding": "20px 0 0",
                        "fontSize": "13px", "color": TEXT_LIGHT,
                        "borderTop": f"1px solid {BORDER}",
                    },
                    children=(
                        f"Loaded {len(DF):,} monitoring stations  ·  "
                        f"{len(TARGET_COLS)} target variables  ·  "
                        "Interpolation: cubic spline over USA bounding box"
                    ),
                ),

            ],
        ),
    ],
)


# ─────────────────────────────────────────────
# CALLBACK: Quick-fill date buttons
# ─────────────────────────────────────────────
@app.callback(
    Output("date-picker", "date"),
    [Input(f"btn-{label.lower().replace(' ', '-')}", "n_clicks")
     for label, _ in DATE_SHORTCUTS],
    prevent_initial_call=True,
)
def fill_date(*btn_clicks):
    """Set the date picker to the corresponding shortcut date when a quick-fill button is clicked."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    # Identify which button fired
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]  # e.g. "btn-today"
    for label, d in DATE_SHORTCUTS:
        if btn_id == f"btn-{label.lower().replace(' ', '-')}":
            return d.isoformat()

    return no_update


# ─────────────────────────────────────────────
# CALLBACK: Run prediction & update map
# ─────────────────────────────────────────────
@app.callback(
    Output("usa-map",    "figure"),
    Output("status-msg", "children"),
    Output("map-title",  "children"),
    Output("stats-panel","children"),
    Input("predict-btn", "n_clicks"),
    State("target-radio","value"),
    State("model-radio", "value"),
    State("date-picker", "date"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, target, model_type, selected_date):
    """
    Validate inputs → predict at all stations → interpolate to grid → render heatmap.
    Returns updated map figure, status message, map title, and summary stats.
    """

    # ── Input validation ───────────────────────
    errors = []
    if not target:
        errors.append("a target variable")
    if not selected_date:
        errors.append("a prediction date")

    if errors:
        # Build a friendly red error banner
        msg = html.Div(
            f"⚠️  Please select {' and '.join(errors)} before running a prediction.",
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

    # ── Predict ────────────────────────────────
    pred_date = date.fromisoformat(selected_date)
    station_df = predict_at_stations(target, model_type, pred_date)

    # ── Interpolate to grid ────────────────────
    lon_grid, lat_grid, val_grid = interpolate_to_grid(station_df)

    # Flatten for Plotly densitymapbox-style heatmap via Scattergeo approach
    # We use a filled contour layer drawn via go.Contour wrapped in a custom geo heatmap
    colorscale = TARGET_COLORSCALES.get(target, "Viridis")
    unit       = TARGET_UNITS.get(target, "")

    vmin = float(np.nanpercentile(val_grid, 2))   # 2nd percentile clips outliers
    vmax = float(np.nanpercentile(val_grid, 98))

    fig = go.Figure()

    # ── Interpolated heatmap layer (grid of scatter dots) ──
    # Densify the grid into flat arrays, masking NaN cells
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()
    flat_vals = val_grid.ravel()
    mask      = ~np.isnan(flat_vals)

    # Downsample for performance (keep ≤ 4 000 points)
    idx = np.where(mask)[0]
    if len(idx) > 4000:
        idx = np.random.default_rng(0).choice(idx, 4000, replace=False)

    fig.add_trace(go.Scattergeo(
        lat=flat_lats[idx],
        lon=flat_lons[idx],
        mode="markers",
        marker=dict(
            size=9,
            color=flat_vals[idx],
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            opacity=0.72,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"{target}<br>({unit})", font=dict(size=13)),
                thickness=14,
                len=0.7,
                x=1.01,
            ),
            line=dict(width=0),
        ),
        name="Predicted grid",
        hovertemplate=f"<b>{target}</b>: %{{marker.color:.2f}} {unit}<br>"
                       "Lat: %{lat:.2f}  Lon: %{lon:.2f}<extra></extra>",
    ))

    # ── Station markers (actual monitoring locations) ────
    fig.add_trace(go.Scattergeo(
        lat=station_df[LAT_COL],
        lon=station_df[LON_COL],
        mode="markers",
        marker=dict(
            size=7,
            color=station_df["predicted"],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            line=dict(color="white", width=0.8),
            symbol="circle",
        ),
        name="Monitoring stations",
        hovertemplate=(
            f"<b>Station — {target}</b><br>"
            f"Predicted: %{{marker.color:.2f}} {unit}<br>"
            "Lat: %{lat:.2f}  Lon: %{lon:.2f}<extra></extra>"
        ),
    ))

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
        height=520,
    )

    # ── Success status message ─────────────────
    status = html.Div(
        f"✅  Prediction complete — {model_type}  ·  {pred_date.strftime('%b %d, %Y')}",
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

    # ── Map title ──────────────────────────────
    title = (
        f"{target} ({unit})  ·  {model_type}  ·  "
        f"Predicted for {pred_date.strftime('%B %d, %Y')}"
    )

    # ── Summary stats panel ────────────────────
    preds  = station_df["predicted"]
    stats  = html.Div(
        style={**CARD_STYLE, "padding": "18px"},
        children=[
            html.Span("Prediction Summary", style=LABEL_STYLE),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                children=[
                    _stat_tile("Min",    f"{preds.min():.2f} {unit}"),
                    _stat_tile("Max",    f"{preds.max():.2f} {unit}"),
                    _stat_tile("Mean",   f"{preds.mean():.2f} {unit}"),
                    _stat_tile("Std Dev",f"{preds.std():.2f} {unit}"),
                ],
            ),
        ],
    )

    return fig, status, title, stats


def _stat_tile(label: str, value: str) -> html.Div:
    """Render a small metric tile for the summary stats panel."""
    return html.Div(
        style={
            "background": BG_PAGE,
            "borderRadius": "8px",
            "padding": "10px 12px",
            "textAlign": "center",
        },
        children=[
            html.Div(label, style={"fontSize": "11px", "color": TEXT_LIGHT,
                                   "fontWeight": "600", "textTransform": "uppercase",
                                   "letterSpacing": "0.05em", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "15px", "fontWeight": "700",
                                   "color": TEXT_DARK}),
        ],
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
