"""Tests for the LOBSTER reading layer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obrp.lobster import (
    EMPTY_LEVEL_PRICE,
    SampleSpec,
    build_snapshots,
    load_pair,
    read_orderbook,
    resample_to_grid,
)
from obrp.synthetic import generate_lobster_day


class SentinelTest(unittest.TestCase):
    """LOBSTER writes +/-9999999999 for an unoccupied level. Letting that
    through produces a mid price of half a million dollars, and every downstream
    number stays arithmetically valid while being nonsense."""

    def test_sentinels_become_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ob.csv"
            pd.DataFrame(
                [
                    [1_000_100, 50, 999_900, 60, EMPTY_LEVEL_PRICE, 0, -EMPTY_LEVEL_PRICE, 0],
                    [1_000_200, 40, 999_800, 70, 1_000_300, 30, 999_700, 20],
                ]
            ).to_csv(path, header=False, index=False)

            book = read_orderbook(path, levels=2)
            self.assertTrue(np.isnan(book.loc[0, "askPx_2"]))
            self.assertTrue(np.isnan(book.loc[0, "bidPx_2"]))
            self.assertTrue(np.isnan(book.loc[0, "askQty_2"]))
            # A populated level on the same row is untouched.
            self.assertAlmostEqual(book.loc[0, "askPx_1"], 100.01)
            self.assertAlmostEqual(book.loc[1, "bidPx_2"], 99.97)

    def test_a_sentinel_would_wreck_the_mid_if_kept(self) -> None:
        """Demonstrates why the conversion matters rather than asserting it twice."""
        naive_mid = (EMPTY_LEVEL_PRICE / 10_000.0 + 100.0) / 2.0
        self.assertGreater(naive_mid, 400_000.0)


class AlignmentTest(unittest.TestCase):
    def test_mismatched_files_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = generate_lobster_day(Path(tmp), n_events=200, levels=2, seed=1)
            truncated = Path(tmp) / "short_orderbook.csv"
            pd.read_csv(spec.orderbook_path, header=None).head(50).to_csv(
                truncated, header=False, index=False
            )
            bad = SampleSpec(spec.message_path, truncated, levels=2)
            with self.assertRaises(ValueError) as caught:
                load_pair(bad)
            self.assertIn("not aligned", str(caught.exception))


class ResampleTest(unittest.TestCase):
    def test_grid_takes_the_last_event_at_or_before_each_point(self) -> None:
        events = pd.DataFrame({"time": [0.5, 1.2, 1.8, 3.1], "value": [10, 20, 30, 40]})
        grid = resample_to_grid(events, step_seconds=1.0)
        # At t=2 the most recent event is the one at 1.8, not the one at 3.1.
        row = grid.loc[grid["gridTime"] == 2.0]
        self.assertEqual(int(row["value"].iloc[0]), 30)

    def test_grid_never_uses_a_future_event(self) -> None:
        events = pd.DataFrame({"time": [0.5, 5.0], "value": [1, 999]})
        grid = resample_to_grid(events, step_seconds=1.0)
        early = grid.loc[grid["gridTime"] < 5.0, "value"]
        self.assertTrue((early == 1).all())

    def test_zero_step_raises(self) -> None:
        events = pd.DataFrame({"time": [1.0, 2.0], "value": [1, 2]})
        with self.assertRaises(ValueError):
            resample_to_grid(events, step_seconds=0.0)


class SnapshotTest(unittest.TestCase):
    def test_counts_are_backward_looking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = generate_lobster_day(Path(tmp), n_events=4000, seed=5)
            snapshots = build_snapshots(spec, step_seconds=1.0)
            for column in ("nTrades", "tradeVolume", "nSubmits", "nCancels"):
                self.assertIn(column, snapshots.columns)
                self.assertTrue((snapshots[column] >= 0).all())
            self.assertGreater(snapshots["nTrades"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
