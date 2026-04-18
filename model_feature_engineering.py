from __future__ import annotations

import numpy as np


def add_derived_features(X: np.ndarray) -> np.ndarray:
    """
    Add compact temporal, climate, and spatial interaction features while
    keeping the raw input contract unchanged.
    """
    X = np.asarray(X, dtype=float)
    doy = X[:, 0]
    season_angle = 2.0 * np.pi * doy / 366.0
    season_angle_2 = 2.0 * season_angle
    high_f = X[:, 2]
    high_c = X[:, 3]
    low_f = X[:, 4]
    low_c = X[:, 5]
    precip = X[:, 6]
    snow = X[:, 7]
    snow_depth = X[:, 8]
    gdd = X[:, 1]
    distance_km = X[:, 9]

    if X.shape[1] >= 12:
        latitude = X[:, 10]
        longitude = X[:, 11]
    else:
        latitude = np.zeros(len(X))
        longitude = np.zeros(len(X))

    frozen = (high_c <= 0.0).astype(float)
    moisture = precip + snow + snow_depth
    derived = np.column_stack([
        np.sin(season_angle),
        np.cos(season_angle),
        np.sin(season_angle_2),
        np.cos(season_angle_2),
        high_f - low_f,
        high_c - low_c,
        (high_c + low_c) / 2.0,
        moisture,
        np.log1p(np.clip(moisture, a_min=0.0, a_max=None)),
        gdd * np.clip(precip, a_min=0.0, a_max=None),
        frozen,
        latitude * np.sin(season_angle),
        longitude * np.cos(season_angle),
        distance_km * np.cos(season_angle),
    ])
    return np.hstack([X, derived])
