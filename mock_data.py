"""Generate synthetic OHLC data with trending and ranging regimes."""

import numpy as np
import pandas as pd


def generate_ohlc(
    n_bars: int = 2000,
    start: str = "2025-01-01",
    freq: str = "1min",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns [open, high, low, close] on a
    DatetimeIndex.

    The price path alternates between up-trend, down-trend, and ranging
    regimes so the market-structure indicator has realistic material.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_bars, freq=freq)

    # --- regime schedule (drift per bar) ---
    regimes = []
    remaining = n_bars
    while remaining > 0:
        length = rng.integers(150, 400)
        length = min(length, remaining)
        kind = rng.choice(["up", "down", "range"])
        drift = {"up": 0.05, "down": -0.05, "range": 0.0}[kind]
        regimes.extend([drift] * length)
        remaining -= length

    drifts = np.array(regimes[:n_bars])
    volatility = 0.3  # per-bar noise scale

    # --- random-walk close prices ---
    returns = drifts + rng.normal(0, volatility, n_bars)
    log_prices = np.cumsum(returns)
    closes = 100.0 * np.exp(log_prices / 100.0)  # keep near 100

    # --- derive OHLC from close ---
    noise = lambda: np.abs(rng.normal(0, 0.15, n_bars))  # noqa: E731
    opens = closes + rng.normal(0, 0.1, n_bars)
    highs = np.maximum(opens, closes) + noise()
    lows = np.minimum(opens, closes) - noise()

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )
