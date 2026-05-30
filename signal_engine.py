"""Shared signal generation primitives."""

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

import pandas as pd

import binance_data
import tradfi_data
from clusters import ALL_CLUSTERS, ALL_TIMEFRAMES, get_cluster
from ms_engine import compute_cluster_signals, get_mtf_trend
from universe import UNIVERSE, Symbol

PIVOT_LENGTH = 2
WARMUP_BARS = 500


@dataclass
class Signal:
    symbol: str
    cluster: str
    direction: str
    timestamp: datetime
    price: float
    source: str = ""


def _source_group_key(symbol: Symbol) -> tuple[str, str]:
    """Group symbols by provider and provider-specific market bucket."""
    return symbol.source, symbol.source_param


def fetch_all(
    universe: list[Symbol], n_bars: int = 2000
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch all timeframes for every symbol, grouped by source."""
    all_data: dict[tuple[str, str], pd.DataFrame] = {}

    sorted_universe = sorted(universe, key=_source_group_key)

    for (source, param), syms in groupby(sorted_universe, key=_source_group_key):
        syms = list(syms)
        tasks = [(s.name, tf) for s in syms for tf in ALL_TIMEFRAMES]

        if source == "binance":
            data = binance_data.fetch_multi(tasks, n_bars=n_bars, market=param)
        else:
            data = tradfi_data.fetch_multi(tasks, n_bars=n_bars, asset_class=param)

        all_data.update(data)

    return all_data


def _compute_cluster_trends(
    df_l: pd.DataFrame,
    df_m: pd.DataFrame,
    df_h: pd.DataFrame,
    cluster: dict[str, str],
    pivot_length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute low, medium, and high timeframe trends on the low timeframe index."""
    trend_l = get_mtf_trend(df_l, cluster["low"], pivot_length, higher_tf_df=df_l)
    trend_m = get_mtf_trend(df_l, cluster["med"], pivot_length, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df_l, cluster["high"], pivot_length, higher_tf_df=df_h)
    return trend_l, trend_m, trend_h


def _trim_cluster_output(
    df: pd.DataFrame,
    longs,
    shorts,
    show_length: int | None,
):
    """Return the requested output window after running the full warmup history."""
    if show_length is None:
        return df, longs, shorts
    return df[-show_length:], longs[-show_length:], shorts[-show_length:]


def run_cluster(
    df: pd.DataFrame,
    df_m: pd.DataFrame,
    df_h: pd.DataFrame,
    cluster: dict[str, str] | None = None,
    pivot_length: int = PIVOT_LENGTH,
    show_length: int | None = None,
):
    """Compute trends and cluster signals from pre-fetched data."""
    cluster = cluster or get_cluster("C1")

    trend_l, trend_m, trend_h = _compute_cluster_trends(
        df,
        df_m,
        df_h,
        cluster,
        pivot_length,
    )
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return _trim_cluster_output(df, longs, shorts, show_length)


def cluster_frames(
    data: dict[tuple[str, str], pd.DataFrame],
    symbol: str,
    cluster: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return low, medium, and high timeframe candles for one symbol+cluster."""
    return (
        data[(symbol, cluster["low"])],
        data[(symbol, cluster["med"])],
        data[(symbol, cluster["high"])],
    )


def signals_from_cluster_output(
    symbol: str,
    source: str,
    cluster_name: str,
    df: pd.DataFrame,
    longs,
    shorts,
    row_indexes=None,
) -> list[Signal]:
    """Convert long/short boolean arrays into domain-level Signal objects."""
    signals: list[Signal] = []
    rows = range(len(df)) if row_indexes is None else row_indexes

    for i in rows:
        ts = df.index[i].to_pydatetime()
        price = float(df["close"].iloc[i])
        if longs[i]:
            signals.append(Signal(symbol, cluster_name, "long", ts, price, source))
        if shorts[i]:
            signals.append(Signal(symbol, cluster_name, "short", ts, price, source))

    return signals


def _run_symbol_cluster(
    sym: Symbol,
    cluster_name: str,
    cluster: dict[str, str],
    data: dict[tuple[str, str], pd.DataFrame],
    show_length: int,
) -> list[Signal]:
    """Run one configured cluster for one symbol and return its signals."""
    df_l, df_m, df_h = cluster_frames(data, sym.name, cluster)
    df, longs, shorts = run_cluster(
        df=df_l,
        df_m=df_m,
        df_h=df_h,
        cluster=cluster,
        show_length=show_length,
    )
    return signals_from_cluster_output(
        sym.name,
        sym.source,
        cluster_name,
        df,
        longs,
        shorts,
    )


def generate_signals(
    universe: list[Symbol] | None = None,
    n_bars: int = 2000,
    show_length: int | None = None,
) -> list[Signal]:
    """Run all clusters on all symbols and return structured signals."""
    universe = universe or UNIVERSE
    output_length = n_bars if show_length is None else show_length
    data = fetch_all(universe, n_bars=n_bars + WARMUP_BARS)
    signals: list[Signal] = []

    for sym in universe:
        for cname, cluster in ALL_CLUSTERS.items():
            signals.extend(
                _run_symbol_cluster(sym, cname, cluster, data, output_length)
            )

    return signals
