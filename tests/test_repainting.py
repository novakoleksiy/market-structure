"""Demonstrate that compute_market_structure repaints when history length changes.

The state machine depends on pivots detected by argrelmax/argrelmin, which are
window-relative.  Prepending bars can change which bars are local extrema,
producing different pivot sequences and therefore different trend values on the
same tail data.
"""

import numpy as np
import pytest

from ms_engine import compute_market_structure


def _make_ohlc(n: int, seed: int = 42):
    """Generate synthetic but reproducible OHLC arrays."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    noise = np.abs(rng.standard_normal(n) * 0.3)
    high = close + noise
    low = close - noise
    return high, low, close


@pytest.mark.parametrize("pivot_length", [2, 3, 5])
def test_trend_changes_with_history_length(pivot_length: int):
    """Trend on the last `tail` bars differs when computed on full vs truncated data."""
    full = 2000
    tail = 500

    high, low, close = _make_ohlc(full)

    # Compute trend on full history, take the tail
    trend_full = compute_market_structure(high, low, close, pivot_length)
    trend_full_tail = trend_full[-tail:]

    # Compute trend on only the tail slice
    trend_short = compute_market_structure(
        high[-tail:], low[-tail:], close[-tail:], pivot_length
    )

    mismatches = np.where(trend_full_tail != trend_short)[0]

    if len(mismatches) == 0:
        pytest.skip("No divergence found with this seed/params (unlikely but possible)")

    print(
        f"\n  pivot_length={pivot_length}: {len(mismatches)} mismatches "
        f"out of {tail} bars ({100 * len(mismatches) / tail:.1f}%)"
    )
    print(
        f"  First mismatch at tail index {mismatches[0]}: "
        f"full={trend_full_tail[mismatches[0]]}, short={trend_short[mismatches[0]]}"
    )


@pytest.mark.parametrize("pivot_length", [2, 3, 5])
def test_trend_stable_with_warmup(pivot_length: int):
    """With sufficient warmup, trends on the output window must be identical."""
    full = 2000
    output = 500
    warmup = 200

    high, low, close = _make_ohlc(full)

    # Run A: use full history, take last `output` bars
    trend_full = compute_market_structure(high, low, close, pivot_length)
    trend_a = trend_full[-output:]

    # Run B: use output + warmup, take last `output` bars
    total_b = output + warmup
    trend_b_full = compute_market_structure(
        high[-total_b:], low[-total_b:], close[-total_b:], pivot_length
    )
    trend_b = trend_b_full[-output:]

    mismatches = np.where(trend_a != trend_b)[0]
    assert len(mismatches) == 0, (
        f"{len(mismatches)} mismatches with {warmup}-bar warmup; "
        f"first at output index {mismatches[0]}"
    )
