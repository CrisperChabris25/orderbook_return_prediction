"""Tests for feature construction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obrp.features import FEATURE_COLUMNS, add_quote_features, build_features
from obrp.lobster import build_snapshots
from obrp.synthetic import generate_lobster_day


class QuoteFeatureTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bidPx_1": [100.00, 100.00],
                "askPx_1": [100.02, 100.02],
                "bidQty_1": [900.0, 100.0],
                "askQty_1": [100.0, 900.0],
            }
        )

    def test_microprice_leans_toward_the_thin_side(self) -> None:
        """A heavy bid queue means the next trade is more likely to lift the
        offer, so fair value sits nearer the ask."""
        out = add_quote_features(self.frame())
        self.assertGreater(out.loc[0, "microprice"], out.loc[0, "mid"])
        self.assertLess(out.loc[1, "microprice"], out.loc[1, "mid"])

    def test_microprice_equals_mid_when_the_book_is_balanced(self) -> None:
        balanced = self.frame()
        balanced["bidQty_1"] = 500.0
        balanced["askQty_1"] = 500.0
        out = add_quote_features(balanced)
        np.testing.assert_allclose(out["microprice"], out["mid"])

    def test_queue_imbalance_is_bounded_and_signed(self) -> None:
        out = add_quote_features(self.frame())
        self.assertAlmostEqual(out.loc[0, "queueImbalance"], 0.8)
        self.assertAlmostEqual(out.loc[1, "queueImbalance"], -0.8)

    def test_spread_in_bps_matches_a_hand_calculation(self) -> None:
        out = add_quote_features(self.frame())
        expected = 0.02 / 100.01 * 10_000.0
        self.assertAlmostEqual(out.loc[0, "spreadBps"], expected, places=6)


class PanelFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        spec = generate_lobster_day(Path(cls._tmp.name), n_events=8000, seed=11)
        cls.frame = build_features(build_snapshots(spec, 1.0), spec.levels)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_declared_feature_exists(self) -> None:
        for feature in FEATURE_COLUMNS:
            self.assertIn(feature, self.frame.columns)

    def test_imbalances_stay_within_minus_one_and_one(self) -> None:
        for column in (
            "queueImbalance",
            "weightedImbalance",
            "depthImbalance_L1",
            "depthImbalance_L10",
            "tradeImbalance",
            "flowImbalance_15",
        ):
            values = self.frame[column].dropna()
            self.assertGreaterEqual(values.min(), -1.0 - 1e-9, column)
            self.assertLessEqual(values.max(), 1.0 + 1e-9, column)

    def test_empty_intervals_are_encoded_not_dropped(self) -> None:
        """Most one-second intervals contain no trade. Those must become a
        neutral zero with an indicator, not a NaN that removes the row."""
        empty = self.frame["tradeVolume"] == 0
        self.assertGreater(empty.sum(), 0, "expected some intervals with no trades")
        self.assertTrue((self.frame.loc[empty, "tradeImbalance"] == 0).all())
        self.assertTrue((self.frame.loc[empty, "hasTrade"] == 0).all())
        self.assertFalse(self.frame.loc[empty, "cancelToTrade"].isna().any())

    def test_deeper_cumulative_imbalance_is_smoother(self) -> None:
        """Aggregating more levels should reduce variance, since level-1 noise
        is averaged against the rest of the book."""
        shallow = self.frame["depthImbalance_L1"].dropna().std()
        deep = self.frame["depthImbalance_L10"].dropna().std()
        self.assertLess(deep, shallow)


if __name__ == "__main__":
    unittest.main()
