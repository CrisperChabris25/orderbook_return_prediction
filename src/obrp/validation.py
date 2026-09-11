"""Walk-forward validation with purging and an embargo.

Three things separate this from a plain time-series split, and all three exist
because of the forward-looking target.

Walk-forward. Folds move forward in time and the model is never fitted on data
that comes after what it is tested on. A shuffled split would be meaningless
here: order book state is strongly autocorrelated, so neighbouring rows are
near-duplicates and random assignment puts copies of the same information on
both sides.

Purging. A target at time t spans [t, t+h]. Training rows within h steps of the
test period therefore have targets computed partly from test-period prices, so
they are dropped. Without this, the model is fitted on the answer.

Embargo. Even after purging, rows immediately after the test period remain
correlated with it through the same persistent features. An embargo drops a
further block so the next training window does not begin flush against data the
model has just been scored on.

The cost is real -- purge and embargo together remove observations from every
fold -- and is reported rather than absorbed silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    purged: int
    embargoed: int

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def walk_forward_folds(
    n_rows: int,
    *,
    n_folds: int = 5,
    horizon: int = 10,
    embargo: int = 0,
    min_train: int = 500,
    expanding: bool = True,
) -> list[Fold]:
    """Folds moving forward in time, purged and embargoed.

    With `expanding` the training window grows and keeps all history; otherwise
    it rolls at a fixed length. Expanding is the default because one trading day
    does not provide enough history to discard any of it.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if embargo < 0:
        raise ValueError("embargo must be nonnegative")

    usable = n_rows - horizon  # rows after this have no complete target
    if usable <= min_train:
        raise ValueError(
            f"Not enough rows: {n_rows} with horizon {horizon} leaves {usable} "
            f"usable against a minimum training size of {min_train}"
        )

    test_size = (usable - min_train) // n_folds
    if test_size < 1:
        raise ValueError(
            f"{n_folds} folds leaves no test rows; reduce n_folds or min_train"
        )

    folds: list[Fold] = []
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = test_start + test_size if fold < n_folds - 1 else usable

        # Purge: a training row at i has a target spanning [i, i+horizon], so it
        # must end strictly before the test period begins.
        train_end = test_start - horizon
        train_start = 0 if expanding else max(0, train_end - min_train)
        if train_end <= train_start:
            continue

        # Embargo: drop the opening block of the test period, which is the part
        # most correlated with the end of training.
        embargoed_start = min(test_start + embargo, test_end)

        folds.append(
            Fold(
                train=np.arange(train_start, train_end),
                test=np.arange(embargoed_start, test_end),
                purged=horizon,
                embargoed=embargoed_start - test_start,
            )
        )
    return folds


def describe_folds(folds: list[Fold]) -> str:
    lines = [f"{'fold':>5}{'train':>10}{'test':>9}{'purged':>9}{'embargoed':>11}"]
    for index, fold in enumerate(folds, start=1):
        lines.append(
            f"{index:>5}{fold.n_train:>10}{fold.n_test:>9}"
            f"{fold.purged:>9}{fold.embargoed:>11}"
        )
    return "\n".join(lines)
