"""Tests for the backtesting engine (backtest.py) and ATR helper."""

from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest
from backtest import (
    Trade,
    backtest_symbol_cluster,
    compute_metrics,
    run_backtest,
)
from ms_engine import compute_atr

_CLUSTER = {"low": "5min", "med": "30min", "high": "4h"}


def _ohlc(rows, freq="5min"):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq)
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=idx
    )


# ---------------------------------------------------------------------------
# compute_atr
# ---------------------------------------------------------------------------


def test_atr_nan_before_period():
    n = 20
    high = np.full(n, 11.0)
    low = np.full(n, 9.0)
    close = np.full(n, 10.0)
    atr = compute_atr(high, low, close, period=14)
    assert np.all(np.isnan(atr[:14]))
    assert not np.isnan(atr[14])


def test_atr_constant_range():
    # Constant high-low range of 2, flat closes -> true range is 2 everywhere,
    # so Wilder's ATR converges to exactly 2.
    n = 30
    high = np.full(n, 11.0)
    low = np.full(n, 9.0)
    close = np.full(n, 10.0)
    atr = compute_atr(high, low, close, period=14)
    assert atr[14] == 2.0
    assert np.isclose(atr[-1], 2.0)


def test_atr_too_short():
    atr = compute_atr(np.ones(5), np.ones(5), np.ones(5), period=14)
    assert len(atr) == 5
    assert np.all(np.isnan(atr))


# ---------------------------------------------------------------------------
# backtest_symbol_cluster — SL / TP / opposite exits
# ---------------------------------------------------------------------------


def _patched_signals(longs, shorts, df):
    """Patch run_cluster to return preset signals on a given frame."""
    return patch.object(
        backtest, "run_cluster", return_value=(df, np.array(longs), np.array(shorts))
    )


def _flat_atr(value, length, period=14):
    """Patch compute_atr to a constant value (nan before `period`)."""
    arr = np.full(length, np.nan)
    arr[period:] = value
    return patch.object(backtest, "compute_atr", return_value=arr)


def test_long_take_profit():
    # Entry at bar 14 (close=100); ATR=2 -> SL=98, TP=104. Bar 15 spikes to TP.
    n = 17
    rows = [(100, 100, 100, 100)] * n
    rows[15] = (100, 105, 100, 100)  # high pierces TP=104
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    shorts = [False] * n
    with _patched_signals(longs, shorts, df), _flat_atr(2.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.exit_reason == "tp"
    assert t.exit_price == 104.0
    assert np.isclose(t.r_multiple, 2.0)
    assert t.bars_held == 1


def test_long_stop_loss():
    n = 17
    rows = [(100, 100, 100, 100)] * n
    rows[15] = (100, 100, 95, 100)  # low pierces SL=98
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    with _patched_signals(longs, [False] * n, df), _flat_atr(2.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert trades[0].exit_reason == "sl"
    assert trades[0].exit_price == 98.0
    assert np.isclose(trades[0].r_multiple, -1.0)


def test_same_bar_sl_first():
    # Bar 15 touches both SL (98) and TP (104) -> stop assumed to fill first.
    n = 17
    rows = [(100, 100, 100, 100)] * n
    rows[15] = (100, 105, 95, 100)
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    with _patched_signals(longs, [False] * n, df), _flat_atr(2.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert trades[0].exit_reason == "sl"


def test_opposite_signal_exit():
    # Long opened at 14; no SL/TP touched; opposite (short) signal at bar 16
    # closes the trade at that bar's close.
    n = 18
    rows = [(100, 100.5, 99.5, 100)] * n  # tight range, never hits SL/TP
    rows[16] = (100, 100.5, 99.5, 101)
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    shorts = [False] * n
    shorts[16] = True
    with _patched_signals(longs, shorts, df), _flat_atr(5.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert trades[0].exit_reason == "opposite"
    assert trades[0].exit_price == 101.0


def test_open_position_at_end():
    n = 17
    rows = [(100, 100.5, 99.5, 100)] * n
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    with _patched_signals(longs, [False] * n, df), _flat_atr(5.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert trades[0].exit_reason == "open"
    assert trades[0].bars_held == n - 1 - 14


def test_no_pyramiding():
    # A second long signal while already long is ignored.
    n = 20
    rows = [(100, 100.5, 99.5, 100)] * n
    df = _ohlc(rows)
    longs = [False] * n
    longs[14] = True
    longs[16] = True  # ignored: still in the first position
    with _patched_signals(longs, [False] * n, df), _flat_atr(5.0, n):
        trades = backtest_symbol_cluster(df, df, df, _CLUSTER, "C1", "X")
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def _trade(r, reason="tp"):
    ts = pd.Timestamp("2024-01-01")
    return Trade(
        symbol="X", cluster="C1", direction="long",
        entry_time=ts, entry_price=100.0, sl=98.0, tp=104.0,
        exit_time=ts, exit_price=100.0, exit_reason=reason,
        r_multiple=r, return_pct=r, bars_held=1,
    )


def test_metrics_basic():
    trades = [_trade(2.0), _trade(2.0), _trade(-1.0, "sl")]
    m = compute_metrics(trades)
    assert m["n_trades"] == 3
    assert np.isclose(m["win_rate"], 2 / 3)
    assert np.isclose(m["total_R"], 3.0)
    assert np.isclose(m["profit_factor"], 4.0)  # 4 won / 1 lost


def test_metrics_excludes_open():
    m = compute_metrics([_trade(2.0), _trade(0.5, "open")])
    assert m["n_trades"] == 1
    assert m["n_open"] == 1


def test_metrics_empty():
    assert compute_metrics([])["n_trades"] == 0


# ---------------------------------------------------------------------------
# run_backtest smoke test (no network)
# ---------------------------------------------------------------------------


def test_run_backtest_smoke():
    from universe import Symbol

    # Build a frame that produces at least one completed long setup on C1.
    n = 400
    base = np.linspace(100, 200, n)
    df = _ohlc([(p, p + 1, p - 1, p) for p in base])
    df_dip = df.copy()
    df_dip.iloc[200:210, :] = df_dip.iloc[200:210, :] - 20  # a dip for structure

    fake_data = {
        ("AAA", tf): df for tf in ("5min", "30min", "4h")
    }
    universe = [Symbol("AAA", "binance", "futures-usdt")]

    with patch.object(backtest, "fetch_all", return_value=fake_data):
        result = run_backtest(universe=universe, clusters=["C1"], n_bars=n)

    assert isinstance(result.metrics, dict)
    assert isinstance(result.equity_curve, pd.Series)
    # Equity curve must be the running sum of closed-trade R-multiples.
    closed = [t for t in result.trades if t.exit_reason != "open"]
    if closed:
        assert np.isclose(
            result.equity_curve.iloc[-1], sum(t.r_multiple for t in closed)
        )


def test_run_backtest_skips_missing_symbol():
    from universe import Symbol

    universe = [Symbol("MISSING", "binance", "futures-usdt")]
    with patch.object(backtest, "fetch_all", return_value={}):
        result = run_backtest(universe=universe, clusters=["C1"], n_bars=100)
    assert result.trades == []
    assert result.metrics["n_trades"] == 0
