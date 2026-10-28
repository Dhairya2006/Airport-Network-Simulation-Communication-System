import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data import make_dataset
from inference import classify, validate_frame
from settings import FEATURES, LABELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class AirportNetTests(unittest.TestCase):
    def setUp(self):
        self.df = make_dataset(sessions=36, observations_per_session=3, seed=123)

    def test_reproducible_synthetic_data(self):
        pd.testing.assert_frame_equal(self.df, make_dataset(sessions=36, observations_per_session=3, seed=123))

    def test_generated_columns_and_ranges(self):
        self.assertTrue(set(FEATURES + ["session_id", "status"]).issubset(self.df.columns))
        self.assertTrue(self.df["status"].isin(LABELS).all())
        self.assertTrue(self.df["utilization_pct"].between(0, 100).all())

    def test_validation_rejects_missing_column(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_frame(self.df.drop(columns=["delay_ms"]))

    def test_validation_rejects_bad_percent(self):
        bad = self.df.iloc[:1].copy()
        bad["packet_loss_pct"] = 110
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            validate_frame(bad)

    def test_smoke_alert_independent_from_classifier(self):
        encoder = ColumnTransformer([("d", OneHotEncoder(handle_unknown="ignore"), ["department"])], remainder="passthrough")
        model = Pipeline([("prep", encoder), ("rf", RandomForestClassifier(n_estimators=10, random_state=1))])
        model.fit(self.df[FEATURES], self.df["status"])
        one = self.df.iloc[:1].copy()
        one["smoke_alert"] = 1
        result = classify(model, one)
        self.assertTrue(result.iloc[0]["alert"].startswith("SMOKE"))
        self.assertIn(result.iloc[0]["predicted_status"], LABELS)


if __name__ == "__main__":
    unittest.main()
