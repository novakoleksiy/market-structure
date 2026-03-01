"""Market structure engine"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import argrelmax, argrelmin


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------


def detect_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    pivot_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of pivot-high and pivot-low values.

    A pivot high at bar *i* is confirmed ``pivot_length`` bars later (bar
    i + pivot_length).  The returned arrays place the value at the
    **confirmation** bar
    """
    n = len(highs)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)

    for i in argrelmax(highs, order=pivot_length)[0]:
        if i + pivot_length < n:
            ph[i + pivot_length] = highs[i]

    for i in argrelmin(lows, order=pivot_length)[0]:
        if i + pivot_length < n:
            pl[i + pivot_length] = lows[i]

    return ph, pl


# ---------------------------------------------------------------------------
# Market-structure state machine (single-bar step)
# ---------------------------------------------------------------------------


@dataclass
class MarketStructureState:
    trend: int = 0
    str_point_hl: float = field(default=float("nan"))
    str_point_lh: float = field(default=float("nan"))
    last_ph: float = field(default=float("nan"))
    last_pl: float = field(default=float("nan"))


def update_market_structure(
    st: MarketStructureState,
    close: float,
    pivot_high: float,
    pivot_low: float,
) -> None:
    """Advance the market-structure state by one bar (mutates *st*)."""
    is_ph = not np.isnan(pivot_high)
    is_pl = not np.isnan(pivot_low)

    signal_ph = is_ph and (not is_pl or st.trend == 1)
    signal_pl = is_pl and (not is_ph or st.trend == -1)

    if signal_ph:
        st.last_ph = pivot_high
    if signal_pl:
        st.last_pl = pivot_low

    if st.trend == 1:
        if close < st.str_point_hl:
            st.trend = -1
            st.str_point_lh = st.last_ph
            st.str_point_hl = float("nan")
        elif signal_pl and (np.isnan(st.str_point_hl) or pivot_low > st.str_point_hl):
            st.str_point_hl = pivot_low
    elif st.trend == -1:
        if close > st.str_point_lh:
            st.trend = 1
            st.str_point_hl = st.last_pl
            st.str_point_lh = float("nan")
        elif signal_ph and (np.isnan(st.str_point_lh) or pivot_high < st.str_point_lh):
            st.str_point_lh = pivot_high
    else:
        if signal_pl:
            st.trend = 1
            st.str_point_hl = pivot_low
        elif signal_ph:
            st.trend = -1
            st.str_point_lh = pivot_high


def compute_market_structure(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    pivot_length: int = 2,
) -> np.ndarray:
    """Return an int array of trend values (1 / -1 / 0) per bar."""
    ph, pl = detect_pivots(highs, lows, pivot_length)
    st = MarketStructureState()
    n = len(closes)
    trend = np.zeros(n, dtype=int)
    for i in range(n):
        update_market_structure(st, closes[i], ph[i], pl[i])
        trend[i] = st.trend
    return trend


# ---------------------------------------------------------------------------
# Multi-timeframe helper
# ---------------------------------------------------------------------------


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a DatetimeIndex OHLC DataFrame to a higher timeframe."""
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def get_mtf_trend(
    df: pd.DataFrame,
    rule: str,
    pivot_length: int = 2,
) -> pd.Series:
    """Compute market-structure trend on a resampled timeframe and
    forward-fill it back onto the base index (mimics ``request.security``
    with ``gaps_off, lookahead_off``).
    """
    resampled = resample_ohlc(df, rule)
    trend_vals = compute_market_structure(
        resampled["high"].values,
        resampled["low"].values,
        resampled["close"].values,
        pivot_length,
    )
    trend_sr = pd.Series(trend_vals, index=resampled.index, name="trend")
    # Reindex onto the base df, forward-fill (no lookahead)
    trend_sr = trend_sr.reindex(df.index, method="ffill").fillna(0).astype(int)
    return trend_sr


# ---------------------------------------------------------------------------
# Cluster signal state machine
# ---------------------------------------------------------------------------


@dataclass
class ClusterState:
    # Long side
    m_dip: bool = False
    m_rec: bool = False
    l_dip: bool = False
    fired: bool = False
    # Short side
    m_dip_s: bool = False
    m_rec_s: bool = False
    l_dip_s: bool = False
    fired_s: bool = False


def _reset_long(cs: ClusterState, t_h: int, t_m: int) -> None:
    if t_h != 1:
        cs.m_dip = cs.m_rec = cs.l_dip = cs.fired = False
    elif cs.m_rec and t_m == -1:
        cs.m_dip = cs.m_rec = cs.l_dip = False
        # fired preserved


def _reset_short(cs: ClusterState, t_h: int, t_m: int) -> None:
    if t_h != -1:
        cs.m_dip_s = cs.m_rec_s = cs.l_dip_s = cs.fired_s = False
    elif cs.m_rec_s and t_m == 1:
        cs.m_dip_s = cs.m_rec_s = cs.l_dip_s = False
        # fired_s preserved


def _advance_long(cs: ClusterState, t_h: int, t_m: int, t_l: int) -> None:
    if t_h == 1 and not cs.fired:
        if t_m == -1:
            cs.m_dip = True
        if cs.m_dip and not cs.m_rec and t_m == 1:
            cs.m_rec = True
            cs.l_dip = False
        if cs.m_rec and t_l == -1:
            cs.l_dip = True


def _advance_short(cs: ClusterState, t_h: int, t_m: int, t_l: int) -> None:
    if t_h == -1 and not cs.fired_s:
        if t_m == 1:
            cs.m_dip_s = True
        if cs.m_dip_s and not cs.m_rec_s and t_m == -1:
            cs.m_rec_s = True
            cs.l_dip_s = False
        if cs.m_rec_s and t_l == 1:
            cs.l_dip_s = True


def compute_cluster_signals(
    trend_h: np.ndarray,
    trend_m: np.ndarray,
    trend_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays (long_signals, short_signals)."""
    n = len(trend_h)
    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    cs = ClusterState()

    for i in range(1, n):
        t_h, t_m, t_l = int(trend_h[i]), int(trend_m[i]), int(trend_l[i])
        t_l_prev = int(trend_l[i - 1])

        # Resets
        _reset_long(cs, t_h, t_m)
        _reset_short(cs, t_h, t_m)

        # Advance setups
        _advance_long(cs, t_h, t_m, t_l)
        _advance_short(cs, t_h, t_m, t_l)

        # Check long signal
        if (
            t_h == 1
            and cs.m_rec
            and cs.l_dip
            and t_l == 1
            and t_l_prev == -1
            and not cs.fired
        ):
            longs[i] = True
            cs.fired = True

        # Check short signal
        if (
            t_h == -1
            and cs.m_rec_s
            and cs.l_dip_s
            and t_l == -1
            and t_l_prev == 1
            and not cs.fired_s
        ):
            shorts[i] = True
            cs.fired_s = True

    return longs, shorts
