"""A synthetic LOBSTER ticker-day, so the repository runs on clone.

LOBSTER data may not be redistributed, so nothing real can ship here. Rather
than provide a separate code path for demos -- which would leave the real
loader untested by anything you can actually execute -- this writes files in
LOBSTER's own format. The message and orderbook readers, the sentinel handling
and the grid resampling are all exercised by the same code that runs on real
data.

The generator is deliberately modest. An efficient price follows a random walk;
the book is built around it with depth decaying by level; empty levels are
written with LOBSTER's sentinel so that handling is exercised too. A small,
genuine predictive relationship is planted between queue imbalance and the next
price move, which is what lets the tests assert the pipeline can find a signal
that is there -- and, with the planting switched off, that it does not report
one that is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .lobster import (
    DIRECTION_ASK,
    DIRECTION_BID,
    EMPTY_LEVEL_PRICE,
    EVENT_CANCEL_PARTIAL,
    EVENT_EXECUTE_VISIBLE,
    EVENT_SUBMIT,
    PRICE_SCALE,
    SampleSpec,
)

OPEN_SECONDS = 34_200.0  # 09:30:00
CLOSE_SECONDS = 57_600.0  # 16:00:00


def generate_lobster_day(
    output_dir: Path,
    *,
    ticker: str = "SYNTH",
    date: str = "2026-01-02",
    levels: int = 10,
    n_events: int = 60_000,
    tick: float = 0.01,
    imbalance_effect: float = 0.35,
    seed: int = 7,
) -> SampleSpec:
    """Write a synthetic message/orderbook pair in LOBSTER format.

    `imbalance_effect` is the size of the planted relationship between queue
    imbalance and the next efficient-price increment, in ticks. Set it to zero
    for a pure random walk with no predictability, which is what the negative
    control test needs.
    """
    rng = np.random.default_rng(seed)

    times = np.sort(rng.uniform(OPEN_SECONDS, CLOSE_SECONDS, n_events))

    # Queue imbalance as a persistent AR(1) process, so it looks like a real
    # book rather than white noise and so trailing features carry information.
    imbalance = np.zeros(n_events)
    for i in range(1, n_events):
        imbalance[i] = 0.97 * imbalance[i - 1] + rng.normal(0, 0.2)
    imbalance = np.clip(imbalance, -1.0, 1.0)

    # The efficient price. The imbalance term is the planted signal: a book
    # leaning bid-heavy nudges the next increment up.
    increments = rng.normal(0, 0.6, n_events) + imbalance_effect * imbalance
    efficient = 100.0 + np.cumsum(increments) * tick

    half_spread = tick * rng.integers(1, 3, n_events)
    best_bid = np.round((efficient - half_spread) / tick) * tick
    best_ask = np.round((efficient + half_spread) / tick) * tick
    best_ask = np.maximum(best_ask, best_bid + tick)

    base_size = rng.lognormal(4.6, 0.7, n_events)
    bid_top = base_size * (1.0 + imbalance)
    ask_top = base_size * (1.0 - imbalance)

    book: dict[str, np.ndarray] = {}
    for level in range(1, levels + 1):
        decay = np.exp(-0.25 * (level - 1))
        # Deep levels are genuinely empty part of the time, which is what puts
        # LOBSTER's sentinel into the file.
        present = rng.random(n_events) > (0.02 * level)

        bid_px = best_bid - (level - 1) * tick
        ask_px = best_ask + (level - 1) * tick
        bid_qty = np.maximum(1.0, bid_top * decay * rng.lognormal(0, 0.3, n_events))
        ask_qty = np.maximum(1.0, ask_top * decay * rng.lognormal(0, 0.3, n_events))

        book[f"bidPx_{level}"] = np.where(present, np.round(bid_px * PRICE_SCALE), -EMPTY_LEVEL_PRICE)
        book[f"bidQty_{level}"] = np.where(present, np.round(bid_qty), 0)
        book[f"askPx_{level}"] = np.where(present, np.round(ask_px * PRICE_SCALE), EMPTY_LEVEL_PRICE)
        book[f"askQty_{level}"] = np.where(present, np.round(ask_qty), 0)

    event_type = rng.choice(
        [EVENT_SUBMIT, EVENT_CANCEL_PARTIAL, EVENT_EXECUTE_VISIBLE],
        size=n_events,
        p=[0.55, 0.30, 0.15],
    )
    direction = rng.choice([DIRECTION_BID, DIRECTION_ASK], size=n_events)
    size = np.round(rng.lognormal(4.0, 0.8, n_events))
    price_int = np.round(np.where(direction == DIRECTION_BID, best_bid, best_ask) * PRICE_SCALE)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{date}_34200000_57600000"
    message_path = output_dir / f"{stem}_message_{levels}.csv"
    orderbook_path = output_dir / f"{stem}_orderbook_{levels}.csv"

    pd.DataFrame(
        {
            "time": times,
            "eventType": event_type,
            "orderId": np.arange(1, n_events + 1),
            "size": size.astype(int),
            "price": price_int.astype(np.int64),
            "direction": direction,
        }
    ).to_csv(message_path, header=False, index=False)

    ordered: dict[str, np.ndarray] = {}
    for level in range(1, levels + 1):
        for field in ("askPx", "askQty", "bidPx", "bidQty"):
            ordered[f"{field}_{level}"] = book[f"{field}_{level}"].astype(np.int64)
    pd.DataFrame(ordered).to_csv(orderbook_path, header=False, index=False)

    return SampleSpec(
        message_path=message_path,
        orderbook_path=orderbook_path,
        levels=levels,
        ticker=ticker,
        date=date,
    )
