"""Models and evaluation.

Two models, chosen to bracket the question rather than to win a bake-off. Ridge
is the linear baseline: if a linear combination of book state carries nothing,
that is worth knowing before reaching for anything larger. Gradient boosting is
the non-linear comparison, deliberately shallow and regularised, because the
sample is one trading day and a deep model would memorise it.

Features are standardised using training-fold statistics only. Fitting a scaler
on the full panel is a quiet form of leakage -- the test period contributes its
own mean and variance to the transform -- and it is easy to do by accident.

Out-of-sample R2 is computed against the TRAINING mean, not the test mean. Using
the test mean would credit the model for knowing the average return of a period
it is supposed to be predicting, which inflates the number and is a real effect
at short horizons where the mean is close to the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass
class FoldResult:
    fold: int
    model: str
    n_train: int
    n_test: int
    r2_in: float
    r2_out: float
    hit_rate: float
    coefficients: dict[str, float] = field(default_factory=dict)


def out_of_sample_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """R2 against the training mean -- the honest benchmark out of sample."""
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - train_mean) ** 2)
    if total <= 0:
        return float("nan")
    return 1.0 - residual / total


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of non-zero predictions whose sign matches the outcome."""
    moved = y_true != 0
    if not moved.any():
        return float("nan")
    return float((np.sign(y_pred[moved]) == np.sign(y_true[moved])).mean())


def build_models(seed: int = 0) -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=10.0),
        "gbm": HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
    }


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using training statistics only."""
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std > 0, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def fit_and_score(
    name: str,
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    fold_index: int,
    feature_names: list[str],
) -> FoldResult:
    x_train_s, x_test_s = standardize(x_train, x_test)
    model.fit(x_train_s, y_train)

    in_pred = model.predict(x_train_s)
    out_pred = model.predict(x_test_s)

    train_mean = float(np.mean(y_train))
    r2_in = out_of_sample_r2(y_train, in_pred, train_mean)
    r2_out = out_of_sample_r2(y_test, out_pred, train_mean)

    coefficients: dict[str, float] = {}
    if hasattr(model, "coef_"):
        coefficients = {
            feature: float(value) for feature, value in zip(feature_names, model.coef_)
        }

    return FoldResult(
        fold=fold_index,
        model=name,
        n_train=len(y_train),
        n_test=len(y_test),
        r2_in=r2_in,
        r2_out=r2_out,
        hit_rate=hit_rate(y_test, out_pred),
        coefficients=coefficients,
    )
