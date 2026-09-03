import unittest

import numpy as np

from model_feature_engineering import add_derived_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_add_derived_features_rejects_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D feature matrix"):
            add_derived_features(np.arange(12))

    def test_add_derived_features_rejects_too_few_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 10 base feature columns"):
            add_derived_features(np.ones((2, 9)))

    def test_add_derived_features_appends_expected_columns(self) -> None:
        base = np.array([[32, 10, 68, 20, 50, 10, 0.5, 0.0, 0.0, 12.0, 41.6, -93.6]])

        engineered = add_derived_features(base)

        self.assertEqual(engineered.shape, (1, 26))
        self.assertEqual(engineered.shape[1] - base.shape[1], 14)
        np.testing.assert_allclose(engineered[:, :12], base)


if __name__ == "__main__":
    unittest.main()
