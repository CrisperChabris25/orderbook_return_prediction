"""Leakage tests.

These are the tests the project exists to pass. Every other result is
meaningless if a feature can see the future, and the failure is silent: a leaky
pipeline reports an out-of-sample R2 that looks like skill and is arithmetic.

The method is to perturb data that a given row must not be able to see, and
assert that nothing about that row changes.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obrp.features import FEATURE_COLUMNS, build_features
from obrp.lobster import build_snapshots
from obrp.synthetic import generate_lobster_day
from obrp.target import forward_return
from obrp.validation import walk_forward_folds


class FeatureLeakageTest(unittest.TestCase):
    """No feature may respond to anything that happens after its timestamp."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        spec = generate_lobster_day(Path(cls._tmp.name), n_events=8000, seed=3)
        cls.levels = spec.levels
        cls.snapshots = build_snapshots(spec, step_seconds=1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_future_rows_cannot_change_past_features(self) -> None:
        """Corrupt the last 40% of the panel beyond recognition. Every feature
        in the first half must be bit-identical."""
        base = build_features(self.snapshots, self.levels)

        corrupted = self.snapshots.copy()
        cut = int(len(corrupted) * 0.6)
        for column in corrupted.columns:
            if column in ("gridTime", "ticker", "date"):
                continue
            if pd.api.types.is_numeric_dtype(corrupted[column]):
                corrupted.loc[corrupted.index[cut:], column] *= 1000.0
        after = build_features(corrupted, self.levels)

        half = cut // 2
        for feature in FEATURE_COLUMNS:
            np.testing.assert_allclose(
                base[feature].to_numpy()[:half],
                after[feature].to_numpy()[:half],
                equal_nan=True,
                err_msg=f"{feature} responded to future data",
            )

    def test_the_test_itself_can_detect_leakage(self) -> None:
        """A guard on the guard. A deliberately leaky feature -- the next row's
        mid -- must be caught by the same perturbation, otherwise the test above
        proves nothing."""
        base = build_features(self.snapshots, self.levels)
        base["leaky"] = base["mid"].shift(-1)

        corrupted = self.snapshots.copy()
        cut = int(len(corrupted) * 0.6)
        corrupted.loc[corrupted.index[cut:], "bidPx_1"] *= 1000.0
        after = build_features(corrupted, self.levels)
        after["leaky"] = after["mid"].shift(-1)

        # The row immediately before the corruption sees it through the leaky
        # feature, and must differ.
        self.assertNotAlmostEqual(
            float(base["leaky"].iloc[cut - 1]), float(after["leaky"].iloc[cut - 1])
        )

    def test_target_looks_forward_and_features_do_not(self) -> None:
        frame = build_features(self.snapshots, self.levels)
        target = forward_return(frame["mid"], horizon=10)
        # The target is null exactly where a full forward window is unavailable.
        self.assertTrue(target.tail(10).isna().all())
        self.assertFalse(target.iloc[:-10].isna().all())


class PurgeAndEmbargoTest(unittest.TestCase):
    HORIZON = 10
    EMBARGO = 25

    def folds(self, n_rows: int = 6000):
        return walk_forward_folds(
            n_rows,
            n_folds=4,
            horizon=self.HORIZON,
            embargo=self.EMBARGO,
            min_train=1000,
        )

    def test_no_training_target_window_reaches_the_test_period(self) -> None:
        """The purge property, stated directly: the last training index plus the
        horizon must land strictly before the first test index."""
        for fold in self.folds():
            last_train = int(fold.train.max())
            first_test = int(fold.test.min())
            self.assertLess(
                last_train + self.HORIZON,
                first_test,
                "a training row's target window overlaps the test period",
            )

    def test_embargo_is_applied(self) -> None:
        for fold in self.folds():
            self.assertEqual(fold.embargoed, self.EMBARGO)

    def test_train_always_precedes_test(self) -> None:
        for fold in self.folds():
            self.assertLess(int(fold.train.max()), int(fold.test.min()))

    def test_folds_move_forward_and_training_expands(self) -> None:
        folds = self.folds()
        for earlier, later in zip(folds, folds[1:]):
            self.assertLess(int(earlier.test.max()), int(later.test.max()))
            self.assertGreater(later.n_train, earlier.n_train)

    def test_test_windows_do_not_overlap(self) -> None:
        folds = self.folds()
        for earlier, later in zip(folds, folds[1:]):
            self.assertLess(int(earlier.test.max()), int(later.test.min()))

    def test_no_row_has_a_target_running_past_the_data(self) -> None:
        n_rows = 6000
        for fold in self.folds(n_rows):
            self.assertLessEqual(int(fold.test.max()) + self.HORIZON, n_rows)

    def test_impossible_configurations_raise(self) -> None:
        with self.assertRaises(ValueError):
            walk_forward_folds(500, n_folds=5, horizon=10, min_train=1000)
        with self.assertRaises(ValueError):
            walk_forward_folds(6000, n_folds=0, horizon=10, min_train=1000)


if __name__ == "__main__":
    unittest.main()
