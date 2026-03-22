"""Entry point: generate mock data, run market-structure engine, plot results."""

from binance_data import fetch_klines
from chart import plot_market_structure
from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
CLUSTER = _CLUSTER_3


def main() -> None:
    symbol = "BTCUSDT"
    market = "futures-usdt"
    limit = 1500

    # 1. Fetch each timeframe directly from Binance
    df = fetch_klines(symbol, CLUSTER["low"], limit=limit, market=market)
    df_m = fetch_klines(symbol, CLUSTER["med"], limit=limit, market=market)
    df_h = fetch_klines(symbol, CLUSTER["high"], limit=limit, market=market)

    # 2. Compute market-structure trend on each timeframe
    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH, higher_tf_df=df)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH, higher_tf_df=df_h)

    # 3. Compute cluster signals
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return df, df_m, longs, shorts


if __name__ == "__main__":
    longs, shorts = main()
