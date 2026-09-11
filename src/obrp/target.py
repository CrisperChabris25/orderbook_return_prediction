"""The prediction target: forward mid-price return.

Defined as the log change in the mid from the current grid point to `horizon`
steps ahead, in basis points. Rows without a full horizon ahead of them are left
null rather than shortened, so no target is ever computed from a partial window.

The forward window is what forces purging in validation. A target at time t
spans [t, t+h], so the last h training rows before a test period carry
information from inside it. Ignoring that is the most common way an order book
model reports out-of-sample skill it does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(mid: pd.Series, horizon: int) -> pd.Series:
    """Log return from t to t+horizon, in basis points, null where incomplete."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 step")
    log_mid = np.log(mid)
    return (log_mid.shift(-horizon) - log_mid) * 10_000.0


def add_target(frame: pd.DataFrame, horizon: int, price_column: str = "mid") -> pd.DataFrame:
    out = frame.copy()
    out[f"fwdRet_{horizon}"] = forward_return(out[price_column], horizon)
    return out
