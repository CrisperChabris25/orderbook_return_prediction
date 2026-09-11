# Order book return prediction

Predicting short-horizon mid-price returns from limit order book state, using
LOBSTER message and orderbook data. The point of the project is the validation
rather than the result: forward-looking targets overlap, order book state is
strongly autocorrelated, and a naive train/test split on this data reports
out-of-sample skill that is arithmetic rather than signal. This implements
walk-forward validation with purging and an embargo, and tests that the pipeline
can tell a real relationship from a random walk.

It runs on clone. No data download is required for the demo.

## Method

**Data.** LOBSTER message and orderbook files, aligned row-wise, resampled onto a
fixed clock grid by taking the last state at or before each grid point. Empty
book levels — LOBSTER writes ±9999999999 — are converted to NaN on read rather
than allowed into a mid price.

**Features (36).** All causal. Level-1 state (spread, microprice deviation, queue
imbalance); book shape across 10 levels (cumulative imbalance at depths 1/3/5/10,
distance-weighted imbalance, log depth per side, book slope); order flow (signed
volume, trade imbalance, cancel-to-trade, trailing flow imbalance over 5/15/60
steps); and dynamics (past returns and realized volatility over the same
windows).

**Target.** Log mid return from *t* to *t+h*, in basis points, null where a full
forward window is unavailable.

**Validation.** Expanding walk-forward. Training rows within *h* steps of a test
period are **purged**, since their targets span into it. A further **embargo**
block is dropped from the start of each test period, because rows adjacent to the
training window stay correlated with it through persistent features. Features are
standardised on training-fold statistics only, and out-of-sample R² is measured
against the **training** mean — using the test mean would credit the model for
knowing the average return of the period it is predicting.

**Models.** Ridge as the linear baseline; shallow gradient boosting as the
non-linear comparison.

## Results

Two synthetic days, 5 folds, ~1,000 test rows per fold, horizon 10 steps.

| Day | Model | R² in-sample | **R² out-of-sample** | Hit rate |
|---|---|---|---|---|
| Imbalance→return signal planted | ridge | 0.408 | **0.349** | 71.2% |
| Imbalance→return signal planted | gbm | 0.621 | **0.325** | 69.8% |
| Pure random walk | ridge | 0.042 | **−0.036** | 49.5% |
| Pure random walk | gbm | 0.385 | **−0.081** | 50.4% |

The bottom two rows are the ones worth reading. On data with no predictability at
all, gradient boosting still achieves an **in-sample R² of 0.385** — it fits the
noise — while out-of-sample it is worse than predicting a constant, and the hit
rate sits at a coin flip. That gap is what purged walk-forward validation exists
to expose, and it is the result this repository is built to demonstrate.

Both configurations are asserted in `tests/test_pipeline.py`, so the numbers
above are regenerated rather than reported from memory.

## What this does not do

- **No real-data result is claimed.** The figures above are from a synthetic
  generator with a known answer. Running on a LOBSTER sample is one command
  (see `scripts/get_lobster_sample.md`), but a single ticker-day is too small to
  support a claim about real markets, and none is made.
- **No transaction costs.** R² and hit rate are statistical measures. A hit rate
  of 70% on a one-tick move does not survive a one-tick spread, and nothing here
  addresses that.
- **One day, one instrument.** No cross-sectional evidence, no regime variation.
- **Gradient boosting overfits** the sample sizes involved, visibly. It is kept
  as the comparison that makes the point, not as a recommended model.

## Running it

```bash
pip install -r requirements.txt

# Synthetic day with a planted signal
PYTHONPATH=src python3 -m obrp.run --synthetic

# Negative control: pure random walk, nothing to find
PYTHONPATH=src python3 -m obrp.run --synthetic --synthetic-effect 0.0
```

On a real LOBSTER sample, see `scripts/get_lobster_sample.md`. LOBSTER data is
not redistributable and is not included; `data/` is gitignored.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -t tests
```

30 tests. The ones that matter:

- **`test_leakage.py`** — corrupts the last 40% of the panel and asserts every
  feature in the first half is unchanged. Includes a guard on the guard: a
  deliberately leaky feature must be *caught* by the same perturbation, so the
  test is shown to have teeth.
- **`test_leakage.py::PurgeAndEmbargoTest`** — asserts the last training index
  plus the horizon lands strictly before the first test index, in every fold.
- **`test_pipeline.py`** — runs end to end on a planted signal and on a random
  walk, and asserts the first is found and the second is not.
