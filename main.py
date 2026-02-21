"""Entry point: generate mock data, run market-structure engine, plot results."""

import matplotlib.pyplot as plt
import numpy as np

from mock_data import generate_ohlc
from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
CLUSTER = {"low": "5min", "med": "30min", "high": "4h"}


def main() -> None:
    # 1. Generate base 1-min OHLC data
    df = generate_ohlc(n_bars=20000, freq="1min", seed=42)

    # 2. Compute market-structure trend on each timeframe
    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH)

    # 3. Compute cluster signals
    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )

    # 4. Plot
    fig, ax = plt.subplots(figsize=(18, 6))

    # Price line
    ax.plot(df.index, df["close"], linewidth=0.6, color="black", label="Close")

    # H-TF trend background coloring
    trend_h_vals = trend_h.values
    for i in range(1, len(df)):
        if trend_h_vals[i] == 1:
            ax.axvspan(df.index[i - 1], df.index[i], alpha=0.08, color="green")
        elif trend_h_vals[i] == -1:
            ax.axvspan(df.index[i - 1], df.index[i], alpha=0.08, color="red")

    # Signal markers
    long_idx = np.where(longs)[0]
    short_idx = np.where(shorts)[0]

    if len(long_idx):
        ax.scatter(
            df.index[long_idx],
            df["close"].values[long_idx],
            marker="^",
            color="lime",
            edgecolors="green",
            s=120,
            zorder=5,
            label="Long",
        )
    if len(short_idx):
        ax.scatter(
            df.index[short_idx],
            df["close"].values[short_idx],
            marker="v",
            color="salmon",
            edgecolors="red",
            s=120,
            zorder=5,
            label="Short",
        )

    ax.set_title(
        f"Market Structure Signals  —  cluster {CLUSTER['high']} / {CLUSTER['med']} / {CLUSTER['low']}"
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.tight_layout()
    plt.show()

    # Summary
    print(f"Total bars : {len(df)}")
    print(f"Long sigs  : {int(longs.sum())}")
    print(f"Short sigs : {int(shorts.sum())}")


if __name__ == "__main__":
    main()
