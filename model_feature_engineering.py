from __future__ import annotations

import numpy as np


def add_derived_features(X: np.ndarray) -> np.ndarray:
    """
    Add a few compact derived signals while keeping the runtime feature input
    contract unchanged.
    """
    X = np.asarray(X, dtype=float)
    doy = X[:, 0]
    season_angle = 2.0 * np.pi * doy / 366.0
    derived = np.column_stack([
        np.sin(season_angle),
        np.cos(season_angle),
        X[:, 2] - X[:, 4],            # high minus low temperature
        X[:, 6] + X[:, 7] + X[:, 8],  # precipitation + snow + snow depth
    ])
    return np.hstack([X, derived])
