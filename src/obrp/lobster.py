"""Reading LOBSTER message and orderbook files into aligned snapshots.

LOBSTER ships two files per ticker-day. The message file is one row per order
book event; the orderbook file is one row per event giving the book state
*after* that event. They are aligned by row, which is what makes the pair usable
without replaying the book yourself.

Sampling. Events arrive irregularly -- thousands in a busy second, none in a
quiet one -- so the raw rows are not a usable index for a forecasting problem.
Everything here is resampled onto a fixed clock grid by taking the last
observation at or before each grid point. That is the state a decision maker
would actually have seen at that instant, and it keeps the target horizon
interpretable in seconds rather than in events.

Empty levels. LOBSTER fills unoccupied book levels with +/-9999999999 and a
size of zero. Those are sentinels, not prices, and are converted to NaN on read
rather than being allowed to propagate into a mid or a spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# LOBSTER's sentinel for an unoccupied level, in price-integer units.
EMPTY_LEVEL_PRICE = 9_999_999_999
PRICE_SCALE = 10_000.0

MESSAGE_COLUMNS = ("time", "eventType", "orderId", "size", "price", "direction")

EVENT_SUBMIT = 1
EVENT_CANCEL_PARTIAL = 2
EVENT_CANCEL_FULL = 3
EVENT_EXECUTE_VISIBLE = 4
EVENT_EXECUTE_HIDDEN = 5
EVENT_HALT = 7

# LOBSTER's direction field is the side of the resting order: 1 = bid, -1 = ask.
# An execution against a resting bid is therefore a seller-initiated trade.
DIRECTION_BID = 1
DIRECTION_ASK = -1


@dataclass(frozen=True)
class SampleSpec:
    """Where a LOBSTER ticker-day lives and how deep it goes."""

    message_path: Path
    orderbook_path: Path
    levels: int
    ticker: str = ""
    date: str = ""


def orderbook_columns(levels: int) -> list[str]:
    """LOBSTER orderbook column order: ask price, ask size, bid price, bid size,
    repeated per level."""
    names: list[str] = []
    for level in range(1, levels + 1):
        names += [
            f"askPx_{level}",
            f"askQty_{level}",
            f"bidPx_{level}",
            f"bidQty_{level}",
        ]
    return names


def _blank_empty_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn LOBSTER's +/-9999999999 sentinels into NaN.

    Skipping this is the classic way to get a mid price of half a million
    dollars on a thinly populated book, and it fails silently because the
    arithmetic is all valid.
    """
    out = frame.copy()
    for column in out.columns:
        if not column.startswith(("askPx_", "bidPx_")):
            continue
        sentinel = out[column].abs() >= EMPTY_LEVEL_PRICE
        out.loc[sentinel, column] = np.nan
        size_column = column.replace("Px_", "Qty_")
        if size_column in out.columns:
            out.loc[sentinel, size_column] = np.nan
    return out


def read_messages(path: Path) -> pd.DataFrame:
    """The message file. Times are seconds after midnight."""
    frame = pd.read_csv(path, header=None, names=list(MESSAGE_COLUMNS))
    frame["price"] = frame["price"] / PRICE_SCALE
    return frame


def read_orderbook(path: Path, levels: int) -> pd.DataFrame:
    """The orderbook file, with sentinels blanked and prices in dollars."""
    frame = pd.read_csv(path, header=None, names=orderbook_columns(levels))
    frame = _blank_empty_levels(frame)
    for column in frame.columns:
        if column.startswith(("askPx_", "bidPx_")):
            frame[column] = frame[column] / PRICE_SCALE
    return frame


def load_pair(spec: SampleSpec) -> pd.DataFrame:
    """Message and orderbook joined row-wise into one event-indexed frame.

    The row alignment is LOBSTER's own guarantee, so a length mismatch means the
    files are not a matched pair and is raised rather than truncated.
    """
    messages = read_messages(spec.message_path)
    book = read_orderbook(spec.orderbook_path, spec.levels)
    if len(messages) != len(book):
        raise ValueError(
            f"Message and orderbook files are not aligned: {len(messages)} vs "
            f"{len(book)} rows. These must be a matched LOBSTER pair."
        )
    joined = pd.concat([messages, book], axis=1)
    joined["ticker"] = spec.ticker
    joined["date"] = spec.date
    return joined


def resample_to_grid(events: pd.DataFrame, step_seconds: float) -> pd.DataFrame:
    """Last observation at or before each grid point.

    Forward-filling from the past is the only direction that keeps the panel
    causal: a grid point takes the most recent state that had already happened,
    never the next one. The first grid points before any event are dropped
    rather than back-filled.
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    working = events.sort_values("time", kind="mergesort").reset_index(drop=True)
    start = float(working["time"].iloc[0])
    end = float(working["time"].iloc[-1])
    grid = np.arange(np.ceil(start / step_seconds) * step_seconds, end, step_seconds)
    if len(grid) == 0:
        raise ValueError("Grid is empty; the sample is shorter than one step")

    positions = np.searchsorted(working["time"].to_numpy(), grid, side="right") - 1
    valid = positions >= 0
    snapshot = working.iloc[positions[valid]].reset_index(drop=True)
    snapshot.insert(0, "gridTime", grid[valid])
    return snapshot


def event_counts_per_grid(
    events: pd.DataFrame, grid_times: np.ndarray, step_seconds: float
) -> pd.DataFrame:
    """Message activity inside each grid interval, by event type.

    Counted over the interval ending at the grid point, so every count is fully
    in the past at the moment the features are read.
    """
    edges = np.concatenate([[grid_times[0] - step_seconds], grid_times])
    bucket = np.searchsorted(edges, events["time"].to_numpy(), side="right") - 1
    inside = (bucket >= 0) & (bucket < len(grid_times))

    frame = pd.DataFrame(
        {
            "bucket": bucket[inside],
            "eventType": events["eventType"].to_numpy()[inside],
            "size": events["size"].to_numpy()[inside],
            "direction": events["direction"].to_numpy()[inside],
        }
    )

    out = pd.DataFrame({"gridTime": grid_times})
    executions = frame["eventType"].isin([EVENT_EXECUTE_VISIBLE, EVENT_EXECUTE_HIDDEN])

    def _tally(mask: pd.Series, column: str, how: str) -> np.ndarray:
        selected = frame.loc[mask]
        if selected.empty:
            return np.zeros(len(grid_times))
        grouped = selected.groupby("bucket")[column]
        series = grouped.sum() if how == "sum" else grouped.size()
        return series.reindex(range(len(grid_times)), fill_value=0).to_numpy(dtype=float)

    out["nTrades"] = _tally(executions, "size", "count")
    out["tradeVolume"] = _tally(executions, "size", "sum")
    # A trade against a resting bid is seller-initiated, so the signed volume
    # takes the opposite sign to LOBSTER's direction field.
    out["signedVolume"] = -_tally(
        executions & (frame["direction"] == DIRECTION_BID), "size", "sum"
    ) + _tally(executions & (frame["direction"] == DIRECTION_ASK), "size", "sum")
    out["nSubmits"] = _tally(frame["eventType"] == EVENT_SUBMIT, "size", "count")
    out["nCancels"] = _tally(
        frame["eventType"].isin([EVENT_CANCEL_PARTIAL, EVENT_CANCEL_FULL]), "size", "count"
    )
    return out


def build_snapshots(
    spec: SampleSpec, step_seconds: float = 1.0
) -> pd.DataFrame:
    """A LOBSTER ticker-day as a causal, fixed-interval snapshot panel."""
    events = load_pair(spec)
    snapshot = resample_to_grid(events, step_seconds)
    counts = event_counts_per_grid(events, snapshot["gridTime"].to_numpy(), step_seconds)
    return snapshot.merge(counts, on="gridTime", how="left")
