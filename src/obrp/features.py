"""Causal features from a snapshot panel.

Every feature here is a function of information available at or before its grid
timestamp. That is not a comment, it is the property the whole exercise depends
on: a single feature that peeks one step ahead produces an out-of-sample R2 that
looks like skill and is arithmetic. tests/test_leakage.py enforces it by
perturbing the future and asserting no feature moves.

Trailing windows use `shift(1)` before rolling wherever the current value would
otherwise be included in its own predictor. Where a window legitimately includes
the current observation -- realized volatility over the last k steps, say -- it
is stated in the docstring rather than left to inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAILING_WINDOWS = (5, 15, 60)
IMBALANCE_DEPTHS = (1, 3, 5, 10)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0, np.nan)


def add_quote_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Level-1 state: mid, spread, microprice and top-of-book imbalance."""
    out = frame.copy()
    bid, ask = out["bidPx_1"], out["askPx_1"]
    bid_qty, ask_qty = out["bidQty_1"], out["askQty_1"]

    out["mid"] = (bid + ask) / 2.0
    out["spread"] = ask - bid
    out["spreadBps"] = _safe_divide(out["spread"], out["mid"]) * 10_000.0

    # Microprice weights each side by the size resting on the OPPOSITE side:
    # a large bid queue means the next trade is more likely to lift the offer,
    # so the fair price sits nearer the ask.
    total_top = bid_qty + ask_qty
    out["microprice"] = _safe_divide(bid * ask_qty + ask * bid_qty, total_top)
    out["micropriceDev"] = _safe_divide(out["microprice"] - out["mid"], out["mid"]) * 10_000.0
    out["queueImbalance"] = _safe_divide(bid_qty - ask_qty, total_top)
    return out


def add_depth_features(frame: pd.DataFrame, levels: int) -> pd.DataFrame:
    """Multi-level book shape: cumulative imbalance, distance-weighted
    imbalance, total depth and book slope.

    Cumulative imbalance at depth k sums the first k levels on each side. The
    weighted version discounts each level by 1/(1 + ticks from the touch), so a
    thousand shares five cents away counts for less than a thousand at the
    touch. Missing levels -- LOBSTER's sentinels, already NaN -- are treated as
    zero depth rather than dropping the row.
    """
    out = frame.copy()
    tick = out["spread"].replace(0, np.nan).abs().mode()
    tick_size = float(tick.iloc[0]) if len(tick) else 0.01

    bid_cum = pd.Series(0.0, index=out.index)
    ask_cum = pd.Series(0.0, index=out.index)
    bid_weighted = pd.Series(0.0, index=out.index)
    ask_weighted = pd.Series(0.0, index=out.index)

    for level in range(1, levels + 1):
        bid_qty = out[f"bidQty_{level}"].fillna(0.0)
        ask_qty = out[f"askQty_{level}"].fillna(0.0)
        bid_cum = bid_cum + bid_qty
        ask_cum = ask_cum + ask_qty

        bid_distance = (out["bidPx_1"] - out[f"bidPx_{level}"]).abs() / max(tick_size, 1e-9)
        ask_distance = (out[f"askPx_{level}"] - out["askPx_1"]).abs() / max(tick_size, 1e-9)
        bid_weighted = bid_weighted + bid_qty / (1.0 + bid_distance.fillna(level - 1))
        ask_weighted = ask_weighted + ask_qty / (1.0 + ask_distance.fillna(level - 1))

        if level in IMBALANCE_DEPTHS:
            out[f"depthImbalance_L{level}"] = _safe_divide(bid_cum - ask_cum, bid_cum + ask_cum)

    out["weightedImbalance"] = _safe_divide(
        bid_weighted - ask_weighted, bid_weighted + ask_weighted
    )
    out["logDepthBid"] = np.log1p(bid_cum)
    out["logDepthAsk"] = np.log1p(ask_cum)
    out["depthRatio"] = np.log1p(bid_cum) - np.log1p(ask_cum)

    # Slope: how fast depth accumulates as you walk away from the touch. A flat
    # book is one that gives way easily.
    top_depth = out["bidQty_1"].fillna(0.0) + out["askQty_1"].fillna(0.0)
    out["bookSlope"] = _safe_divide(bid_cum + ask_cum, top_depth)
    return out


def add_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Order flow over the interval ending at each grid point, plus trailing
    aggregates.

    The per-interval counts are already backward-looking by construction (see
    lobster.event_counts_per_grid). The trailing sums shift by one step first,
    so a row's trailing window never contains its own interval.
    """
    out = frame.copy()

    # An interval with no trades has no imbalance to measure. Encoding that as
    # NaN would be correct in principle and disastrous in practice: on a
    # one-second grid most intervals are empty, and dropping them costs about
    # 90% of the panel. It is encoded as zero -- the neutral value, meaning no
    # directional pressure observed -- with an explicit indicator so the model
    # can still tell "balanced flow" apart from "no flow at all".
    out["hasTrade"] = (out["tradeVolume"] > 0).astype(float)
    out["tradeImbalance"] = _safe_divide(out["signedVolume"], out["tradeVolume"]).fillna(0.0)

    # Smoothed rather than guarded, since nTrades is zero in most intervals.
    out["cancelToTrade"] = out["nCancels"] / (out["nTrades"] + 1.0)
    out["logTradeVolume"] = np.log1p(out["tradeVolume"])
    out["orderIntensity"] = out["nSubmits"] + out["nCancels"]

    for window in TRAILING_WINDOWS:
        shifted_signed = out["signedVolume"].shift(1)
        shifted_volume = out["tradeVolume"].shift(1)
        out[f"signedVolume_{window}"] = shifted_signed.rolling(window, min_periods=window).sum()
        out[f"tradeVolume_{window}"] = shifted_volume.rolling(window, min_periods=window).sum()
        out[f"flowImbalance_{window}"] = _safe_divide(
            out[f"signedVolume_{window}"], out[f"tradeVolume_{window}"]
        ).fillna(0.0)
        out[f"cancelRate_{window}"] = (
            out["nCancels"].shift(1).rolling(window, min_periods=window).mean()
        )
    return out


def add_dynamics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Past returns and realized volatility.

    Returns are log mid changes in basis points. The k-step trailing return ends
    at the current row, so it uses the current mid and k-step-old mid -- both
    already observed. Realized volatility is the root sum of squared one-step
    returns over the trailing window, again ending at the current row.
    """
    out = frame.copy()
    log_mid = np.log(out["mid"])
    out["ret1"] = (log_mid - log_mid.shift(1)) * 10_000.0

    for window in TRAILING_WINDOWS:
        out[f"pastReturn_{window}"] = (log_mid - log_mid.shift(window)) * 10_000.0
        out[f"realizedVol_{window}"] = np.sqrt(
            (out["ret1"] ** 2).rolling(window, min_periods=window).sum()
        )
        out[f"imbalanceMean_{window}"] = (
            out["queueImbalance"].shift(1).rolling(window, min_periods=window).mean()
        )
    out["spreadBpsChange"] = out["spreadBps"] - out["spreadBps"].shift(1)
    return out


FEATURE_COLUMNS: tuple[str, ...] = (
    # level-1 state
    "spreadBps",
    "micropriceDev",
    "queueImbalance",
    # book shape
    "depthImbalance_L1",
    "depthImbalance_L3",
    "depthImbalance_L5",
    "depthImbalance_L10",
    "weightedImbalance",
    "logDepthBid",
    "logDepthAsk",
    "depthRatio",
    "bookSlope",
    # flow
    "hasTrade",
    "tradeImbalance",
    "cancelToTrade",
    "logTradeVolume",
    "orderIntensity",
    "flowImbalance_5",
    "flowImbalance_15",
    "flowImbalance_60",
    "tradeVolume_5",
    "tradeVolume_15",
    "tradeVolume_60",
    "cancelRate_5",
    "cancelRate_15",
    "cancelRate_60",
    # dynamics
    "ret1",
    "pastReturn_5",
    "pastReturn_15",
    "pastReturn_60",
    "realizedVol_5",
    "realizedVol_15",
    "realizedVol_60",
    "imbalanceMean_5",
    "imbalanceMean_15",
    "imbalanceMean_60",
    "spreadBpsChange",
)


def build_features(snapshots: pd.DataFrame, levels: int) -> pd.DataFrame:
    """The full causal feature panel."""
    out = add_quote_features(snapshots)
    out = add_depth_features(out, levels)
    out = add_flow_features(out)
    out = add_dynamics_features(out)
    return out
