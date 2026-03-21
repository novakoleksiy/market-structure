"""Entry point: generate mock data, run market-structure engine, plot results."""

import pandas as pd

from binance_data import fetch_klines
from ms_engine import compute_cluster_signals, get_mtf_trend, resample_ohlc

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
CLUSTER = _CLUSTER_3


def main() -> None:
    df = fetch_klines("BTCUSDT", CLUSTER["low"], limit=1000, market="futures-usdt")

    # 2. Compute market-structure trend on each timeframe
    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH)

    # 3. Compute cluster signals
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return longs, shorts


if __name__ == "__main__":
    longs, shorts = main()
