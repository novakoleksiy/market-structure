"""Tests for ms_engine.py — targeting 100% coverage."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ms_engine import (
    ClusterState,
    MarketStructureState,
    _advance_long,
    _advance_short,
    _reset_long,
    _reset_short,
    compute_cluster_signals,
    compute_market_structure,
    detect_pivots,
    get_mtf_trend,
    resample_ohlc,
    update_market_structure,
)

NAN = float("nan")


def _ohlc_df(rows: list[tuple], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# ---------------------------------------------------------------------------
# detect_pivots
# ---------------------------------------------------------------------------


def test_detect_pivots_simple_high():
    highs = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
    lows = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ph, pl = detect_pivots(highs, lows, pivot_length=1)
    assert ph[3] == 5.0


def test_detect_pivots_simple_low():
    lows = np.array([5.0, 4.0, 1.0, 4.0, 5.0])
    highs = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
    ph, pl = detect_pivots(highs, lows, pivot_length=1)
    assert pl[3] == 1.0


def test_detect_pivots_end_not_confirmed():
    highs = np.array([1.0, 2.0, 5.0])
    lows = np.array([1.0, 1.0, 1.0])
    ph, pl = detect_pivots(highs, lows, pivot_length=1)
    assert np.all(np.isnan(ph))


def test_detect_pivots_no_pivots_monotonic():
    highs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lows = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    ph, pl = detect_pivots(highs, lows, pivot_length=1)
    assert np.all(np.isnan(ph))
    assert np.all(np.isnan(pl))


def test_detect_pivots_length_2():
    highs = np.array([1.0, 2.0, 3.0, 10.0, 3.0, 2.0, 1.0])
    lows = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ph, pl = detect_pivots(highs, lows, pivot_length=2)
    assert ph[5] == 10.0


# ---------------------------------------------------------------------------
# update_market_structure
# ---------------------------------------------------------------------------


def test_ums_initial_state_is_neutral():
    st = MarketStructureState()
    assert st.trend == 0


def test_ums_first_pivot_low_starts_uptrend():
    st = MarketStructureState()
    update_market_structure(st, 100.0, NAN, 90.0)
    assert st.trend == 1
    assert st.support == 90.0


def test_ums_first_pivot_high_starts_downtrend():
    st = MarketStructureState()
    update_market_structure(st, 100.0, 110.0, NAN)
    assert st.trend == -1
    assert st.resistance == 110.0


def test_ums_both_pivots_neutral_no_change():
    st = MarketStructureState()
    update_market_structure(st, 100.0, 110.0, 90.0)
    assert st.trend == 0


def test_ums_uptrend_break_support_flips_down():
    st = MarketStructureState(trend=1, support=95.0, last_ph=110.0)
    update_market_structure(st, 90.0, NAN, NAN)
    assert st.trend == -1
    assert st.resistance == 110.0
    assert np.isnan(st.support)


def test_ums_uptrend_higher_low_updates_support():
    st = MarketStructureState(trend=1, support=90.0)
    update_market_structure(st, 100.0, NAN, 95.0)
    assert st.support == 95.0


def test_ums_uptrend_lower_low_keeps_support():
    st = MarketStructureState(trend=1, support=95.0)
    update_market_structure(st, 100.0, NAN, 85.0)
    assert st.support == 95.0


def test_ums_uptrend_nan_support_sets_first_low():
    st = MarketStructureState(trend=1, support=NAN)
    update_market_structure(st, 100.0, NAN, 92.0)
    assert st.support == 92.0


def test_ums_downtrend_break_resistance_flips_up():
    st = MarketStructureState(trend=-1, resistance=110.0, last_pl=85.0)
    update_market_structure(st, 115.0, NAN, NAN)
    assert st.trend == 1
    assert st.support == 85.0
    assert np.isnan(st.resistance)


def test_ums_downtrend_lower_high_updates_resistance():
    st = MarketStructureState(trend=-1, resistance=110.0)
    update_market_structure(st, 100.0, 105.0, NAN)
    assert st.resistance == 105.0


def test_ums_downtrend_higher_high_keeps_resistance():
    st = MarketStructureState(trend=-1, resistance=100.0)
    update_market_structure(st, 95.0, 105.0, NAN)
    assert st.resistance == 100.0


def test_ums_downtrend_nan_resistance_sets_first_high():
    st = MarketStructureState(trend=-1, resistance=NAN)
    update_market_structure(st, 95.0, 105.0, NAN)
    assert st.resistance == 105.0


def test_ums_both_pivots_uptrend_prefers_ph():
    st = MarketStructureState(trend=1, support=90.0)
    update_market_structure(st, 100.0, 115.0, 92.0)
    assert st.last_ph == 115.0


def test_ums_both_pivots_downtrend_prefers_pl():
    st = MarketStructureState(trend=-1, resistance=110.0)
    update_market_structure(st, 100.0, 108.0, 88.0)
    assert st.last_pl == 88.0


# ---------------------------------------------------------------------------
# compute_market_structure
# ---------------------------------------------------------------------------


def test_cms_returns_correct_length_and_dtype():
    highs = np.array([10.0, 12.0, 15.0, 12.0, 10.0, 8.0, 5.0])
    lows = np.array([8.0, 10.0, 13.0, 10.0, 8.0, 6.0, 3.0])
    closes = np.array([9.0, 11.0, 14.0, 11.0, 9.0, 7.0, 4.0])
    trend = compute_market_structure(highs, lows, closes, pivot_length=1)
    assert len(trend) == len(highs)
    assert trend.dtype == int


def test_cms_values_are_valid():
    np.random.seed(42)
    n = 50
    closes = np.cumsum(np.random.randn(n)) + 100
    highs = closes + np.abs(np.random.randn(n))
    lows = closes - np.abs(np.random.randn(n))
    trend = compute_market_structure(highs, lows, closes)
    assert set(trend).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# resample_ohlc
# ---------------------------------------------------------------------------


def test_resample_5min_to_30min():
    rows = [(100, 105, 95, 102)] * 30
    df = _ohlc_df(rows, freq="5min")
    resampled = resample_ohlc(df, "30min")
    assert len(resampled) == 5
    assert list(resampled.columns) == ["open", "high", "low", "close"]


def test_resample_preserves_ohlc_semantics():
    rows = [
        (100, 110, 90, 105),
        (105, 120, 85, 115),
        (115, 130, 80, 125),
    ]
    df = _ohlc_df(rows, freq="5min")
    resampled = resample_ohlc(df, "15min")
    assert len(resampled) == 1
    row = resampled.iloc[0]
    assert row["open"] == 100
    assert row["high"] == 130
    assert row["low"] == 80
    assert row["close"] == 125


# ---------------------------------------------------------------------------
# get_mtf_trend
# ---------------------------------------------------------------------------


def test_mtf_aligned_to_base_index():
    rows = [(100, 105, 95, 102)] * 60
    df = _ohlc_df(rows, freq="5min")
    trend = get_mtf_trend(df, "30min", pivot_length=2)
    assert isinstance(trend, pd.Series)
    assert len(trend) == len(df)
    assert trend.index.equals(df.index)


def test_mtf_values_are_valid():
    rows = [(100, 105, 95, 102)] * 12
    df = _ohlc_df(rows, freq="5min")
    trend = get_mtf_trend(df, "30min", pivot_length=2)
    assert set(trend.values).issubset({-1, 0, 1})


def test_mtf_trend_not_available_before_bar_closes():
    """Higher-TF trend must only appear after the HTF bar closes (lookahead_off).

    Setup: 18 base bars (5min) = three 30min HTF bars.
    Mock compute_market_structure to return [0, 0, 1] for the 3 HTF bars.
    The trend=1 from HTF bar 2 (open at 01:00) should NOT appear on base bars
    01:00–01:25 — it should only appear from 01:30 onward (after bar 2 closes).
    """
    base_rows = [(100, 105, 95, 102)] * 18
    df = _ohlc_df(base_rows, freq="5min")

    htf_rows = [(100, 105, 95, 102)] * 3
    htf = _ohlc_df(htf_rows, freq="30min")

    with patch("ms_engine.compute_market_structure", return_value=np.array([0, 0, 1])):
        trend = get_mtf_trend(df, "30min", pivot_length=2, higher_tf_df=htf)

    # Bars 0–5 (00:00–00:25): HTF bar 0 hasn't closed yet → should be 0
    assert all(trend.iloc[:6] == 0), (
        f"Base bars within HTF bar 0 should show 0, got {trend.iloc[:6].values}"
    )
    # Bars 6–11 (00:30–00:55): HTF bar 0 closed with trend=0 → should be 0
    assert all(trend.iloc[6:12] == 0), (
        f"Base bars within HTF bar 1 should show 0 (from closed bar 0), got {trend.iloc[6:12].values}"
    )
    # Bars 12–17 (01:00–01:25): HTF bar 1 closed with trend=0 → should be 0
    # HTF bar 2 (trend=1) has NOT closed yet, so it must NOT be visible here
    assert all(trend.iloc[12:] == 0), (
        f"Base bars within HTF bar 2 should still show 0 (bar 2 not closed), "
        f"got {trend.iloc[12:].values}"
    )


def test_mtf_trend_shift_applies_to_low_tf():
    """Even the low TF trend should be delayed by one bar (lookahead_off).

    When higher_tf_df is the base df itself, each bar's trend should only
    be visible from the NEXT bar onward.
    """
    base_rows = [(100, 105, 95, 102)] * 6
    df = _ohlc_df(base_rows, freq="5min")

    mock_trends = np.array([0, 0, 1, 1, 1, -1])
    with patch("ms_engine.compute_market_structure", return_value=mock_trends):
        trend = get_mtf_trend(df, "5min", pivot_length=2, higher_tf_df=df)

    # Bar 0: no prior bar → 0
    assert trend.iloc[0] == 0
    # Bar 1: sees bar 0's trend (0)
    assert trend.iloc[1] == 0
    # Bar 2: sees bar 1's trend (0), NOT bar 2's trend (1)
    assert trend.iloc[2] == 0, (
        f"Bar 2 should show bar 1's trend (0), not its own (1), got {trend.iloc[2]}"
    )
    # Bar 3: sees bar 2's trend (1)
    assert trend.iloc[3] == 1


def test_get_mtf_trend_with_higher_tf_df_delays_by_one_bar():
    """Passing higher_tf_df explicitly (as main.py does) must still delay."""
    base_rows = [(100, 105, 95, 102)] * 12
    df = _ohlc_df(base_rows, freq="5min")

    htf_rows = [(100, 105, 95, 102)] * 2
    htf = _ohlc_df(htf_rows, freq="30min")

    with patch("ms_engine.compute_market_structure", return_value=np.array([0, 1])):
        trend = get_mtf_trend(df, "30min", pivot_length=2, higher_tf_df=htf)

    # First 6 base bars (HTF bar 0 period): no prior HTF bar → 0
    assert all(trend.iloc[:6] == 0)
    # Next 6 base bars (HTF bar 1 period): HTF bar 0 trend (0) forward-filled
    assert all(trend.iloc[6:12] == 0), (
        f"HTF bar 1 period should show bar 0's trend (0), got {trend.iloc[6:12].values}"
    )


def test_cluster_signal_not_premature_with_lookahead_fix():
    """Integration test: a signal must not fire during an unclosed HTF bar.

    Scenario: 36 base bars (5min) = three 1h HTF bars (12 base bars each).
    HTF trend flips from -1 to 1 on bar 1. Without the fix, t_h=1 is visible
    from base bar 12 onward (HTF bar 1's open), and the medium dip/recovery +
    low dip/recovery sequence fires a premature long signal at base bar 21.
    With the fix, t_h stays -1 until base bar 24 (after HTF bar 1 closes),
    preventing the premature signal.
    """
    n = 36  # 3 HTF bars × 12 base bars each
    base_rows = [(100, 105, 95, 102)] * n
    df = _ohlc_df(base_rows, freq="5min")

    # High TF (1h): 3 bars, trend flips on bar 1
    htf_rows = [(100, 105, 95, 102)] * 3
    htf = _ohlc_df(htf_rows, freq="1h")
    htf_trends = np.array([-1, 1, 1])

    with patch("ms_engine.compute_market_structure", return_value=htf_trends):
        trend_h = get_mtf_trend(df, "1h", 2, higher_tf_df=htf)

    # Manually construct med and low trends on the base index.
    # Med: dips during early HTF bar 1, recovers from base bar 16.
    trend_m = np.zeros(n, dtype=int)
    trend_m[:16] = -1
    trend_m[16:] = 1

    # Low: dips at base bars 18–20, recovers at 21 → would signal at 21
    # if t_h=1 were visible (medium already recovered, low just flipped).
    trend_l = np.ones(n, dtype=int)
    trend_l[18:21] = -1

    longs, _ = compute_cluster_signals(
        trend_h.values, trend_m, trend_l
    )

    # With the fix: t_h=1 is NOT visible until base bar 24, so the setup
    # that builds during bars 12–23 never activates → no signal at bar 21.
    assert not longs[12:24].any(), (
        f"No long signal should fire during unclosed HTF bar 1, "
        f"got signals at {np.where(longs[12:24])[0] + 12}"
    )


# ---------------------------------------------------------------------------
# _reset_long / _reset_short
# ---------------------------------------------------------------------------


def test_reset_long_when_high_not_up():
    cs = ClusterState(m_dip=True, m_rec=True, l_dip=True, fired=True)
    _reset_long(cs, t_h=0, t_m=1)
    assert not cs.m_dip and not cs.m_rec and not cs.l_dip and not cs.fired


def test_reset_long_partial_m_rec_and_medium_down():
    cs = ClusterState(m_dip=True, m_rec=True, l_dip=True, fired=True)
    _reset_long(cs, t_h=1, t_m=-1)
    assert not cs.m_dip and not cs.m_rec and not cs.l_dip
    assert cs.fired  # preserved


def test_reset_long_no_reset_m_rec_false():
    cs = ClusterState(m_dip=True, m_rec=False)
    _reset_long(cs, t_h=1, t_m=-1)
    assert cs.m_dip


def test_reset_long_no_reset_high_up_medium_up():
    cs = ClusterState(m_dip=True, m_rec=True, l_dip=True)
    _reset_long(cs, t_h=1, t_m=1)
    assert cs.m_dip and cs.m_rec and cs.l_dip


def test_reset_short_when_high_not_down():
    cs = ClusterState(m_dip_s=True, m_rec_s=True, l_dip_s=True, fired_s=True)
    _reset_short(cs, t_h=0, t_m=-1)
    assert not cs.m_dip_s and not cs.m_rec_s and not cs.l_dip_s and not cs.fired_s


def test_reset_short_partial_m_rec_s_and_medium_up():
    cs = ClusterState(m_dip_s=True, m_rec_s=True, l_dip_s=True, fired_s=True)
    _reset_short(cs, t_h=-1, t_m=1)
    assert not cs.m_dip_s and not cs.m_rec_s and not cs.l_dip_s
    assert cs.fired_s


def test_reset_short_no_reset_m_rec_s_false():
    cs = ClusterState(m_dip_s=True, m_rec_s=False)
    _reset_short(cs, t_h=-1, t_m=1)
    assert cs.m_dip_s


def test_reset_short_no_reset_high_down_medium_down():
    cs = ClusterState(m_dip_s=True, m_rec_s=True, l_dip_s=True)
    _reset_short(cs, t_h=-1, t_m=-1)
    assert cs.m_dip_s and cs.m_rec_s and cs.l_dip_s


# ---------------------------------------------------------------------------
# _advance_long / _advance_short
# ---------------------------------------------------------------------------


def test_advance_long_medium_dip():
    cs = ClusterState()
    _advance_long(cs, t_h=1, t_m=-1, t_l=0)
    assert cs.m_dip


def test_advance_long_recovery_after_dip():
    cs = ClusterState(m_dip=True)
    _advance_long(cs, t_h=1, t_m=1, t_l=0)
    assert cs.m_rec
    assert not cs.l_dip


def test_advance_long_low_dip_after_recovery():
    cs = ClusterState(m_dip=True, m_rec=True)
    _advance_long(cs, t_h=1, t_m=1, t_l=-1)
    assert cs.l_dip


def test_advance_long_no_advance_when_fired():
    cs = ClusterState(fired=True)
    _advance_long(cs, t_h=1, t_m=-1, t_l=-1)
    assert not cs.m_dip


def test_advance_long_no_advance_when_high_not_up():
    cs = ClusterState()
    _advance_long(cs, t_h=-1, t_m=-1, t_l=-1)
    assert not cs.m_dip


def test_advance_short_medium_dip():
    cs = ClusterState()
    _advance_short(cs, t_h=-1, t_m=1, t_l=0)
    assert cs.m_dip_s


def test_advance_short_recovery_after_dip():
    cs = ClusterState(m_dip_s=True)
    _advance_short(cs, t_h=-1, t_m=-1, t_l=0)
    assert cs.m_rec_s
    assert not cs.l_dip_s


def test_advance_short_low_dip_after_recovery():
    cs = ClusterState(m_dip_s=True, m_rec_s=True)
    _advance_short(cs, t_h=-1, t_m=-1, t_l=1)
    assert cs.l_dip_s


def test_advance_short_no_advance_when_fired_s():
    cs = ClusterState(fired_s=True)
    _advance_short(cs, t_h=-1, t_m=1, t_l=1)
    assert not cs.m_dip_s


def test_advance_short_no_advance_when_high_not_down():
    cs = ClusterState()
    _advance_short(cs, t_h=1, t_m=1, t_l=1)
    assert not cs.m_dip_s


# ---------------------------------------------------------------------------
# compute_cluster_signals
# ---------------------------------------------------------------------------


def test_cluster_shape_and_dtype():
    n = 10
    t_h = np.ones(n, dtype=int)
    t_m = np.zeros(n, dtype=int)
    t_l = np.zeros(n, dtype=int)
    longs, shorts = compute_cluster_signals(t_h, t_m, t_l)
    assert longs.shape == (n,)
    assert longs.dtype == bool


def test_cluster_long_signal_full_sequence():
    t_h = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    t_m = np.array([1, -1, -1, 1, 1, 1, 1, 1])
    t_l = np.array([1, 1, 1, 1, -1, -1, 1, 1])
    longs, shorts = compute_cluster_signals(t_h, t_m, t_l)
    assert longs[6]
    assert not shorts.any()


def test_cluster_short_signal_full_sequence():
    t_h = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
    t_m = np.array([-1, 1, 1, -1, -1, -1, -1, -1])
    t_l = np.array([-1, -1, -1, -1, 1, 1, -1, -1])
    longs, shorts = compute_cluster_signals(t_h, t_m, t_l)
    assert shorts[6]
    assert not longs.any()


def test_cluster_fires_only_once():
    t_h = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    t_m = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1])
    t_l = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1])
    longs, _ = compute_cluster_signals(t_h, t_m, t_l)
    assert longs.sum() == 1


def test_cluster_empty_input():
    longs, shorts = compute_cluster_signals(
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    assert len(longs) == 0


def test_cluster_single_bar():
    longs, shorts = compute_cluster_signals(
        np.array([1]), np.array([1]), np.array([1])
    )
    assert not longs[0]


def test_cluster_reset_allows_new_long_signal():
    t_h = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1])
    t_m = np.array([1, -1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, 1, 1])
    t_l = np.array([1, 1, 1, 1, -1, -1, 1, 0, 1, 1, 1, 1, -1, -1, 1])
    longs, _ = compute_cluster_signals(t_h, t_m, t_l)
    assert longs[6]
    assert longs[14]


def test_cluster_reset_allows_new_short_signal():
    t_h = np.array([-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1])
    t_m = np.array([-1, 1, 1, -1, -1, -1, -1, 0, -1, 1, 1, -1, -1, -1, -1])
    t_l = np.array([-1, -1, -1, -1, 1, 1, -1, 0, -1, -1, -1, -1, 1, 1, -1])
    _, shorts = compute_cluster_signals(t_h, t_m, t_l)
    assert shorts[6]
    assert shorts[14]
