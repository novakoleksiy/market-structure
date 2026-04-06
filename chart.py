"""Candlestick charting with market-structure overlays."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ms_engine import compute_market_structure, detect_pivots


def _trend_style(fill: str, border: str) -> dict:
    return dict(
        increasing_fillcolor=fill,
        increasing_line_color=border,
        decreasing_fillcolor=fill,
        decreasing_line_color=border,
    )


_STYLES = {
    1: _trend_style("#80cbc4", "#00897b"),  # light teal fill, bright teal border
    -1: _trend_style("#ff8a80", "#e53935"),  # light red fill, bright red border
    0: _trend_style("#888888", "#888888"),
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

    # String labels collapse weekend/holiday gaps on the categorical x-axis
    x_labels = df.index.strftime("%m-%d %H:%M").to_numpy()

    fig = go.Figure()

    # Candlesticks colored by trend
    for t_val in (1, -1, 0):
        mask = trend == t_val
        if not mask.any():
            continue
        fig.add_trace(
            go.Candlestick(
                x=x_labels[mask],
                open=df["open"].values[mask],
                high=highs[mask],
                low=lows[mask],
                close=closes[mask],
                name=_NAMES[t_val],
                **_STYLES[t_val],
            )
        )

    # Pivot markers – offset scales with actual candle size, not price level
    offset = np.nanmean(highs - lows) * 0.5
    ph_idx = np.where(~np.isnan(ph))[0]
    pl_idx = np.where(~np.isnan(pl))[0]
    ph_actual = ph_idx - pivot_length
    pl_actual = pl_idx - pivot_length

    fig.add_trace(
        go.Scatter(
            x=x_labels[ph_actual],
            y=highs[ph_actual] + offset,
            mode="markers",
            name="Pivot High",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels[pl_actual],
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
                x=x_labels[idx],
                y=lows[idx] - offset * 2,
                mode="markers",
                name="Long Signal",
                marker=dict(symbol="diamond", color="#00e676", size=10),
            )
        )
    if shorts is not None and shorts.any():
        idx = np.where(shorts)[0]
        fig.add_trace(
            go.Scatter(
                x=x_labels[idx],
                y=highs[idx] + offset * 2,
                mode="markers",
                name="Short Signal",
                marker=dict(symbol="diamond", color="#ff1744", size=10),
            )
        )

    symbol = df.attrs.get("symbol")
    timeframe = df.attrs.get("timeframe")
    title_parts = [p for p in (symbol, timeframe) if p]
    fig.update_layout(
        title=dict(text=" · ".join(title_parts)) if title_parts else None,
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=x_labels,
            rangeslider=dict(visible=False),
            showgrid=False,
            linecolor="rgba(255,255,255,0.3)",
        ),
        template="plotly_dark",
        yaxis=dict(
            autorange=True,
            fixedrange=False,
            showgrid=False,
            linecolor="rgba(255,255,255,0.3)",
        ),
    )
    return fig
