import numpy as np
import pandas as pd

import signal_engine


def _ohlc(rows: list[tuple], freq: str = "30min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=idx
    )


def test_run_cluster_uses_unshifted_low_tf_trend(monkeypatch):
    df_l = _ohlc([(1, 2, 0, 1)] * 4)
    df_m = _ohlc([(1, 2, 0, 1)] * 4, freq="4h")
    df_h = _ohlc([(1, 2, 0, 1)] * 4, freq="1D")
    raw_low = np.array([1, 1, -1, 1])
    med = pd.Series([1, 1, 1, 1], index=df_l.index)
    high = pd.Series([1, 1, 1, 1], index=df_l.index)
    captured = {}

    monkeypatch.setattr(
        signal_engine,
        "compute_market_structure",
        lambda highs, lows, closes, pivot_length: raw_low,
    )

    def fake_get_mtf_trend(df, rule, pivot_length, higher_tf_df=None):
        return med if rule == "4h" else high

    monkeypatch.setattr(signal_engine, "get_mtf_trend", fake_get_mtf_trend)

    def fake_compute_cluster_signals(trend_h, trend_m, trend_l):
        captured["trend_l"] = trend_l.copy()
        captured["trend_m"] = trend_m.copy()
        captured["trend_h"] = trend_h.copy()
        return np.zeros(len(trend_l), dtype=bool), np.zeros(len(trend_l), dtype=bool)

    monkeypatch.setattr(
        signal_engine, "compute_cluster_signals", fake_compute_cluster_signals
    )

    signal_engine.run_cluster(
        df_l,
        df_m,
        df_h,
        cluster={"low": "30min", "med": "4h", "high": "1D"},
    )

    np.testing.assert_array_equal(captured["trend_l"], raw_low)
    np.testing.assert_array_equal(captured["trend_m"], med.values)
    np.testing.assert_array_equal(captured["trend_h"], high.values)
