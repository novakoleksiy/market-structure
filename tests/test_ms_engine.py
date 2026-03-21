"""Tests for ms_engine.py — targeting 100% coverage."""

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
