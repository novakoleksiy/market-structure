"""Test that cluster signals are deterministic regardless of limit parameter.

Fetches live Binance data with limit=1000 and limit=1500, both using a warm-up
buffer, and asserts that signals on overlapping bars are identical.
"""

import numpy as np
import pytest

from binance_data import fetch_klines_full
from ms_engine import compute_cluster_signals, get_mtf_trend

SYMBOL = "BTCUSDT"
MARKET = "futures-usdt"
PIVOT_LENGTH = 2
WARMUP = 500
CLUSTER = {"low": "4h", "med": "1D", "high": "1W"}


def _run_with_limit(limit: int):
    """Run the full pipeline for a given output limit, return df index + signals."""
    total = limit + WARMUP
    df = fetch_klines_full(SYMBOL, CLUSTER["low"], n_bars=total, market=MARKET)
    df_m = fetch_klines_full(SYMBOL, CLUSTER["med"], n_bars=total, market=MARKET)
    df_h = fetch_klines_full(SYMBOL, CLUSTER["high"], n_bars=total, market=MARKET)

    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH, higher_tf_df=df)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH, higher_tf_df=df_h)

    # Run cluster signals on full history (including warmup) so the state
    # machine is properly initialised before the output window.
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )

    # Trim to last `limit` bars
    return df.index[-limit:], longs[-limit:], shorts[-limit:]


@pytest.mark.network
def test_signals_deterministic_across_limits():
    """Signals on overlapping bars must be identical for limit=1000 vs limit=1500."""
    idx_a, longs_a, shorts_a = _run_with_limit(1000)
    idx_b, longs_b, shorts_b = _run_with_limit(1500)

    # Find overlapping timestamps (the last 1000 bars of the 1500 run)
    overlap = idx_a.intersection(idx_b)
    assert len(overlap) > 0, "No overlapping bars found"

    # Map overlap back to positions in each result
    mask_a = idx_a.isin(overlap)
    mask_b = idx_b.isin(overlap)

    longs_overlap_a = longs_a[mask_a]
    longs_overlap_b = longs_b[mask_b]
    shorts_overlap_a = shorts_a[mask_a]
    shorts_overlap_b = shorts_b[mask_b]

    long_mismatches = np.where(longs_overlap_a != longs_overlap_b)[0]
    short_mismatches = np.where(shorts_overlap_a != shorts_overlap_b)[0]

    if len(long_mismatches) > 0 or len(short_mismatches) > 0:
        overlap_dates = overlap.sort_values()
        msg_parts = []
        if len(long_mismatches) > 0:
            dates = [str(overlap_dates[i]) for i in long_mismatches]
            msg_parts.append(f"Long mismatches at: {dates}")
        if len(short_mismatches) > 0:
            dates = [str(overlap_dates[i]) for i in short_mismatches]
            msg_parts.append(f"Short mismatches at: {dates}")
        pytest.fail("; ".join(msg_parts))

    print(f"\n  Overlapping bars: {len(overlap)}")
    print(
        f"  Longs in 1000-run: {longs_a.sum()}, in 1500-run overlap: {longs_overlap_b.sum()}"
    )
    print(
        f"  Shorts in 1000-run: {shorts_a.sum()}, in 1500-run overlap: {shorts_overlap_b.sum()}"
    )
    print("  All signals match.")
