"""Entry point: generate mock data, run market-structure engine, plot results."""

from binance_data import fetch_klines, fetch_klines_full
from chart import plot_market_structure
from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
WARMUP_BARS = 500
_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
CLUSTER = _CLUSTER_3


def run_cluster(
    symbol: str = "BTCUSDT",
    market: str = "futures-usdt",
    cluster: dict | None = None,
    limit: int = 1000,
    warmup: int = WARMUP_BARS,
    pivot_length: int = PIVOT_LENGTH,
):
    """Fetch data, compute trends with warm-up buffer, return trimmed results."""
    cluster = cluster or CLUSTER

    # 1. Fetch each timeframe with extra warm-up bars
    total = limit + warmup
    df = fetch_klines_full(symbol, cluster["low"], n_bars=total, market=market)
    df_m = fetch_klines_full(symbol, cluster["med"], n_bars=total, market=market)
    df_h = fetch_klines_full(symbol, cluster["high"], n_bars=total, market=market)

    # 2. Compute trends on full history (including warm-up)
    trend_l = get_mtf_trend(df, cluster["low"], pivot_length, higher_tf_df=df)
    trend_m = get_mtf_trend(df, cluster["med"], pivot_length, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, cluster["high"], pivot_length, higher_tf_df=df_h)

    # 3. Trim to the last `limit` bars of the base timeframe
    df = df.iloc[-limit:]
    trend_l = trend_l.iloc[-limit:]
    trend_m = trend_m.iloc[-limit:]
    trend_h = trend_h.iloc[-limit:]

    # 4. Compute cluster signals on the trimmed window
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return df, df_m, longs, shorts


if __name__ == "__main__":
    longs, shorts = main()
