"""Entry point: generate mock data, run market-structure engine, plot results."""

import pandas as pd

from binance_data import fetch_multi
from chart import plot_market_structure
from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
WARMUP_BARS = 500
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
CLUSTER = _CLUSTER_1


def run_cluster(
    df: pd.DataFrame,
    df_m: pd.DataFrame,
    df_h: pd.DataFrame,
    cluster: dict | None = None,
    pivot_length: int = PIVOT_LENGTH,
    show_length: int = 500,
):
    """Compute trends and cluster signals from pre-fetched data."""
    cluster = cluster or CLUSTER

    # Compute trends on full history (including warm-up)
    trend_l = get_mtf_trend(df, cluster["low"], pivot_length, higher_tf_df=df)
    trend_m = get_mtf_trend(df, cluster["med"], pivot_length, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, cluster["high"], pivot_length, higher_tf_df=df_h)

    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return df[-show_length:], longs[-show_length:], shorts[-show_length:]


if __name__ == "__main__":
    cluster = CLUSTER

    # Build task list: all symbols × cluster timeframes
    tasks = [
        (sym, tf)
        for sym in SYMBOLS
        for tf in [cluster["low"], cluster["med"], cluster["high"]]
    ]

    # Fetch all data in parallel
    data = fetch_multi(tasks, n_bars=2000, market="futures-usdt")

    # Run cluster analysis per symbol
    for symbol in SYMBOLS:
        df, longs, shorts = run_cluster(
            df=data[(symbol, cluster["low"])],
            df_m=data[(symbol, cluster["med"])],
            df_h=data[(symbol, cluster["high"])],
            cluster=cluster,
            show_length=500,
        )
        fig = plot_market_structure(df, longs=longs, shorts=shorts)
        fig.show()
