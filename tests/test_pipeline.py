"""End-to-end tests with a known answer.

A validation framework that cannot distinguish a real signal from noise is
decoration. These run the whole pipeline twice: once on a synthetic day with a
genuine relationship planted between queue imbalance and the next price move,
and once on a pure random walk. The first must be found; the second must not be
reported.

The second is the one that matters. In-sample R2 will be positive in both cases
-- a flexible model always fits noise -- so it is the out-of-sample number,
after purging and embargo, that has to tell them apart.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obrp.run import build_panel, run_experiment, summarize
from obrp.synthetic import generate_lobster_day

HORIZON = 10


def run_day(effect: float, seed: int):
    with tempfile.TemporaryDirectory() as tmp:
        spec = generate_lobster_day(
            Path(tmp), n_events=60_000, imbalance_effect=effect, seed=seed
        )
        panel = build_panel(spec, step_seconds=1.0, horizon=HORIZON)
    results, _ = run_experiment(
        panel, horizon=HORIZON, n_folds=3, embargo=50, min_train=1500, seed=seed
    )
    frame = summarize(results)
    return frame.groupby("model")[["r2InSample", "r2OutOfSample", "hitRate"]].mean()


class PlantedSignalTest(unittest.TestCase):
    """Imbalance genuinely moves the next price. Both models must find it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scores = run_day(effect=0.35, seed=3)

    def test_out_of_sample_r2_is_clearly_positive(self) -> None:
        for model in ("ridge", "gbm"):
            self.assertGreater(self.scores.loc[model, "r2OutOfSample"], 0.05, model)

    def test_hit_rate_beats_a_coin_flip(self) -> None:
        for model in ("ridge", "gbm"):
            self.assertGreater(self.scores.loc[model, "hitRate"], 0.55, model)


class NoSignalTest(unittest.TestCase):
    """A pure random walk. Nothing may be reported."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scores = run_day(effect=0.0, seed=4)

    def test_out_of_sample_r2_is_not_positive(self) -> None:
        for model in ("ridge", "gbm"):
            self.assertLess(self.scores.loc[model, "r2OutOfSample"], 0.01, model)

    def test_hit_rate_is_a_coin_flip(self) -> None:
        for model in ("ridge", "gbm"):
            self.assertAlmostEqual(self.scores.loc[model, "hitRate"], 0.5, delta=0.03, msg=model)

    def test_in_sample_fit_is_still_positive(self) -> None:
        """The point of the exercise. The flexible model fits noise in sample;
        only the purged out-of-sample split exposes it."""
        self.assertGreater(self.scores.loc["gbm", "r2InSample"], 0.1)
        self.assertLess(self.scores.loc["gbm", "r2OutOfSample"], 0.0)


if __name__ == "__main__":
    unittest.main()
