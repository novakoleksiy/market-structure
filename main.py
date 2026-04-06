"""Entry point: multi-source signal generator across Binance & OANDA."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

import pandas as pd

import binance_data
import tradfi_data
from chart import plot_market_structure
from ms_engine import compute_cluster_signals, get_mtf_trend

# -- Config -----------------------------------------------------------------
PIVOT_LENGTH = 2
WARMUP_BARS = 500

_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
_CLUSTER_4 = {"low": "1D", "med": "1W", "high": "1ME"}
ALL_CLUSTERS = {"C1": _CLUSTER_1, "C2": _CLUSTER_2, "C3": _CLUSTER_3, "C4": _CLUSTER_4}
ALL_TIMEFRAMES = list({tf for c in ALL_CLUSTERS.values() for tf in c.values()})


@dataclass(frozen=True)
class Symbol:
    name: str
    source: str  # "binance" or "oanda"
    source_param: str  # "futures-usdt" | "forex" | "indices" | "commodities"


UNIVERSE = [
    Symbol("BTCUSDT", "binance", "futures-usdt"),
    Symbol("ETHUSDT", "binance", "futures-usdt"),
    Symbol("SOLUSDT", "binance", "futures-usdt"),
    Symbol("EUR_USD", "oanda", "forex"),
    Symbol("GBP_USD", "oanda", "forex"),
    Symbol("SPX500_USD", "oanda", "indices"),
    Symbol("XAU_USD", "oanda", "commodities"),
    Symbol("XAG_USD", "oanda", "commodities"),
]


@dataclass
class Signal:
    symbol: str
    cluster: str  # "C1", "C2", "C3"
    direction: str  # "long" or "short"
    timestamp: datetime
    price: float


# -- Data fetching ----------------------------------------------------------


def fetch_all(
    universe: list[Symbol], n_bars: int = 2000
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch all timeframes for every symbol, grouped by source."""
    all_data: dict[tuple[str, str], pd.DataFrame] = {}

    def keyfunc(symbol: Symbol) -> tuple[str, str]:
        return symbol.source, symbol.source_param

    sorted_universe = sorted(universe, key=keyfunc)

    for (source, param), syms in groupby(sorted_universe, key=keyfunc):
        syms = list(syms)
        tasks = [(s.name, tf) for s in syms for tf in ALL_TIMEFRAMES]

        if source == "binance":
            data = binance_data.fetch_multi(tasks, n_bars=n_bars, market=param)
        else:
            data = tradfi_data.fetch_multi(tasks, n_bars=n_bars, asset_class=param)

        all_data.update(data)

    return all_data


# -- Cluster analysis -------------------------------------------------------


def run_cluster(
    df: pd.DataFrame,
    df_m: pd.DataFrame,
    df_h: pd.DataFrame,
    cluster: dict | None = None,
    pivot_length: int = PIVOT_LENGTH,
    show_length: int = 500,
):
    """Compute trends and cluster signals from pre-fetched data."""
    cluster = cluster or _CLUSTER_1

    trend_l = get_mtf_trend(df, cluster["low"], pivot_length, higher_tf_df=df)
    trend_m = get_mtf_trend(df, cluster["med"], pivot_length, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, cluster["high"], pivot_length, higher_tf_df=df_h)

    longs, shorts = compute_cluster_signals(
        trend_h.values, trend_m.values, trend_l.values
    )
    return df[-show_length:], longs[-show_length:], shorts[-show_length:]


# -- Signal generation ------------------------------------------------------


def generate_signals(
    universe: list[Symbol] | None = None,
    n_bars: int = 2000,
    show_length: int = 500,
) -> list[Signal]:
    """Run all clusters on all symbols and return structured signals."""
    universe = universe or UNIVERSE
    data = fetch_all(universe, n_bars=n_bars)
    signals: list[Signal] = []

    for sym in universe:
        for cname, cluster in ALL_CLUSTERS.items():
            df_l = data[(sym.name, cluster["low"])]
            df_m = data[(sym.name, cluster["med"])]
            df_h = data[(sym.name, cluster["high"])]

            df, longs, shorts = run_cluster(
                df=df_l,
                df_m=df_m,
                df_h=df_h,
                cluster=cluster,
                show_length=show_length,
            )

            for i in range(len(df)):
                ts = df.index[i].to_pydatetime()
                price = df["close"].iloc[i]
                if longs[i]:
                    signals.append(Signal(sym.name, cname, "long", ts, price))
                if shorts[i]:
                    signals.append(Signal(sym.name, cname, "short", ts, price))

    return signals


# -- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market structure signal generator")
    parser.add_argument("--chart", action="store_true", help="Show Plotly charts")
    args = parser.parse_args()

    signals = generate_signals()

    for sig in signals:
        print(
            f"[{sig.cluster}] {sig.direction.upper():5s} {sig.symbol:12s} "
            f"@ {sig.price:.5f}  ({sig.timestamp})"
        )
    print(f"\n{len(signals)} signals total")

    if args.chart:
        data = fetch_all(UNIVERSE)
        cluster = _CLUSTER_1
        for sym in UNIVERSE:
            df, longs, shorts = run_cluster(
                df=data[(sym.name, cluster["low"])],
                df_m=data[(sym.name, cluster["med"])],
                df_h=data[(sym.name, cluster["high"])],
                cluster=cluster,
                show_length=500,
            )
            fig = plot_market_structure(df, longs=longs, shorts=shorts)
            fig.show()
