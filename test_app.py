import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import app


def _collect_component_ids(component) -> set[str]:
    ids = set()
    component_id = getattr(component, "id", None)
    if component_id:
        ids.add(component_id)

    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.update(_collect_component_ids(child))
    elif hasattr(children, "children"):
        ids.update(_collect_component_ids(children))

    return ids


class DashboardSmokeTests(unittest.TestCase):
    @staticmethod
    def _first_available_selection() -> tuple[str, str]:
        for target in app.TARGET_COLS:
            for option in app._available_model_types(target):
                if not option.get("disabled"):
                    return target, option["value"]
        raise AssertionError("No target/model combination is available for smoke testing.")

    def test_layout_exposes_core_panels(self) -> None:
        ids = _collect_component_ids(app.app.layout)
        self.assertTrue({"usa-map", "hover-panel", "streak-panel", "stats-panel"}.issubset(ids))

    def test_prediction_smoke(self) -> None:
        target, model_type = self._first_available_selection()
        fig, status, title, subtitle, streak, stats, performance = app.run_prediction(
            1,
            target,
            model_type,
            "2026-05-01",
        )

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(status, "")
        self.assertIn(target, title)
        self.assertIn(model_type, subtitle)
        self.assertGreaterEqual(len(fig.data), 2)

        for component in (streak, stats, performance):
            self.assertTrue(hasattr(component, "to_plotly_json"))

    def test_prediction_rejects_invalid_date_without_crashing(self) -> None:
        target, model_type = self._first_available_selection()
        fig, status, title, subtitle, streak, stats, performance = app.run_prediction(
            1,
            target,
            model_type,
            "not-a-date",
        )

        self.assertIsInstance(fig, go.Figure)
        self.assertIn("a valid prediction date", status.children)
        self.assertIs(title, app.no_update)
        self.assertIs(subtitle, app.no_update)

        for component in (streak, performance):
            self.assertTrue(hasattr(component, "to_plotly_json"))

    def test_model_options_fall_back_to_available_model(self) -> None:
        target, current_model = self._first_available_selection()
        options, selected = app.update_model_options(target, "Unavailable Model")
        available = [option["value"] for option in options if not option.get("disabled")]

        self.assertIn(current_model, available)
        self.assertIn(selected, available)

    def test_build_feature_matrix_preserves_feature_contract(self) -> None:
        feature_matrix = app.build_feature_matrix(app.date(2026, 2, 14))

        self.assertEqual(feature_matrix.columns.tolist(), app.FEATURE_COLS)
        self.assertTrue((feature_matrix["doy"] == 45).all())
        self.assertEqual(len(feature_matrix), len(app.STATIONS))

    def test_build_feature_matrix_fills_all_missing_feature_column(self) -> None:
        stations = app.STATIONS.copy()
        stations["precip"] = np.nan

        with patch.object(app, "STATIONS", stations):
            feature_matrix = app.build_feature_matrix(app.date(2026, 2, 14))

        self.assertFalse(feature_matrix.isna().any().any())
        self.assertTrue((feature_matrix["precip"] == 0.0).all())

    def test_target_assessment_missing_value(self) -> None:
        assessment = app._target_assessment("Nitrate", None)

        self.assertEqual(assessment["label"], "Unknown")
        self.assertEqual(assessment["color"], app.TEXT_MID)

    def test_metric_lookup_handles_missing_columns(self) -> None:
        with patch.object(app, "MODEL_METRICS", pd.DataFrame({"target": ["pH"]})):
            self.assertIsNone(app._get_metric_row("pH", "Gradient Boosting"))
            self.assertTrue(hasattr(app._performance_panel("pH", "Gradient Boosting"), "to_plotly_json"))


if __name__ == "__main__":
    unittest.main()
