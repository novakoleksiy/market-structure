"""Candlestick charting with market-structure overlays."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ms_engine import compute_market_structure, detect_pivots


def _trend_style(color: str) -> dict:
    return dict(
        increasing_fillcolor=color,
        increasing_line_color=color,
        decreasing_fillcolor=color,
        decreasing_line_color=color,
    )


_STYLES = {
    1: _trend_style("#26a69a"),
    -1: _trend_style("#ef5350"),
    0: _trend_style("#888888"),
}
_NAMES = {1: "Uptrend", -1: "Downtrend", 0: "Neutral"}


def plot_market_structure(
    df: pd.DataFrame,
    pivot_length: int = 2,
    longs: np.ndarray | None = None,
    shorts: np.ndarray | None = None,
) -> go.Figure:
    """Plot candlesticks colored by trend with pivot markers and optional cluster signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    pivot_length : int
        Bars required to confirm a pivot.
    longs, shorts : np.ndarray, optional
        Boolean arrays of cluster signal locations.
    """
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    trend = compute_market_structure(highs, lows, closes, pivot_length)
    ph, pl = detect_pivots(highs, lows, pivot_length)

    fig = go.Figure()

    # Candlesticks colored by trend
    for t_val in (1, -1, 0):
        mask = trend == t_val
        if not mask.any():
            continue
        fig.add_trace(
            go.Candlestick(
                x=df.index[mask],
                open=df["open"].values[mask],
                high=highs[mask],
                low=lows[mask],
                close=closes[mask],
                name=_NAMES[t_val],
                **_STYLES[t_val],
            )
        )

    # Pivot markers
    offset = closes.mean() * 0.01
    ph_idx = np.where(~np.isnan(ph))[0]
    pl_idx = np.where(~np.isnan(pl))[0]
    ph_actual = ph_idx - pivot_length
    pl_actual = pl_idx - pivot_length

    fig.add_trace(
        go.Scatter(
            x=df.index[ph_actual],
            y=highs[ph_actual] + offset,
            mode="markers",
            name="Pivot High",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[pl_actual],
            y=lows[pl_actual] - offset,
            mode="markers",
            name="Pivot Low",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    # Cluster signals
    if longs is not None and longs.any():
        idx = np.where(longs)[0]
        fig.add_trace(
            go.Scatter(
                x=df.index[idx],
                y=lows[idx] - offset * 2,
                mode="markers",
                name="Long Signal",
                marker=dict(symbol="arrow-up", color="#00e676", size=12),
            )
        )
    if shorts is not None and shorts.any():
        idx = np.where(shorts)[0]
        fig.add_trace(
            go.Scatter(
                x=df.index[idx],
                y=highs[idx] + offset * 2,
                mode="markers",
                name="Short Signal",
                marker=dict(symbol="arrow-down", color="#ff1744", size=12),
            )
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        yaxis=dict(autorange=True, fixedrange=False),
    )
    return fig
