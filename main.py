"""Entry point: generate mock data, run market-structure engine, plot results."""

import pandas as pd

from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
CLUSTER = {"low": "5min", "med": "30min", "high": "4h"}


def main() -> None:
    df = pd.read_pickle("btcusd.pkl").tail(100)

    # 2. Compute market-structure trend on each timeframe
    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH)

    # 3. Compute cluster signals
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )


if __name__ == "__main__":
    longs, shorts = main()
