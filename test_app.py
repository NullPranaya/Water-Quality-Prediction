import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
