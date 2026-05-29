"""Source-aware signal generation for scheduled runs."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import binance_data
import tradfi_data
from clusters import get_cluster
from signal_engine import WARMUP_BARS, Signal, run_cluster
from signal_store import SignalStore, StoredSignal
from universe import UNIVERSE, Symbol


@dataclass(frozen=True)
class ScheduledRunResult:
    source: str
    cluster: str
    latest_bars: dict[str, datetime]
    signals: list[Signal]

    @property
    def latest_bar(self) -> datetime | None:
        """Return the newest bar across all symbols for summary purposes."""
        if not self.latest_bars:
            return None
        return max(self.latest_bars.values())


def filter_universe_by_source(
    source: str, universe: list[Symbol] | None = None
) -> list[Symbol]:
    """Return symbols from one provider source."""
    universe = UNIVERSE if universe is None else universe
    filtered = [sym for sym in universe if sym.source == source]
    if not filtered:
        raise ValueError(f"No symbols configured for source '{source}'")
    return filtered


def fetch_source_cluster_data(
    source: str,
    cluster_name: str,
    universe: list[Symbol] | None = None,
    n_bars: int = 2000,
    use_cache: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch only the timeframes needed for one source+cluster job."""
    syms = filter_universe_by_source(source, universe)
    cluster = get_cluster(cluster_name)
    tasks = [(sym.name, tf) for sym in syms for tf in cluster.values()]

    source_param = syms[0].source_param
    if any(sym.source_param != source_param for sym in syms):
        data: dict[tuple[str, str], pd.DataFrame] = {}
        for asset_group in sorted({sym.source_param for sym in syms}):
            group = [sym for sym in syms if sym.source_param == asset_group]
            group_tasks = [(sym.name, tf) for sym in group for tf in cluster.values()]
            if source == "binance":
                fetched = binance_data.fetch_multi(
                    group_tasks,
                    n_bars=n_bars,
                    market=asset_group,
                    use_cache=use_cache,
                )
            else:
                fetched = tradfi_data.fetch_multi(
                    group_tasks,
                    n_bars=n_bars,
                    asset_class=asset_group,
                    use_cache=use_cache,
                )
            data.update(fetched)
        return data

    if source == "binance":
        return binance_data.fetch_multi(
            tasks,
            n_bars=n_bars,
            market=source_param,
            use_cache=use_cache,
        )
    if source == "oanda":
        return tradfi_data.fetch_multi(
            tasks,
            n_bars=n_bars,
            asset_class=source_param,
            use_cache=use_cache,
        )
    raise ValueError(f"Unknown source '{source}'. Choose from: ['binance', 'oanda']")


def generate_source_cluster_signals(
    source: str,
    cluster_name: str,
    universe: list[Symbol] | None = None,
    n_bars: int = 2000,
    latest_only: bool = True,
    use_cache: bool = True,
) -> ScheduledRunResult:
    """Generate signals for a single provider source and cluster."""
    syms = filter_universe_by_source(source, universe)
    cluster = get_cluster(cluster_name)
    data = fetch_source_cluster_data(
        source,
        cluster_name,
        universe=syms,
        n_bars=n_bars + WARMUP_BARS,
        use_cache=use_cache,
    )

    signals: list[Signal] = []
    latest_bars: dict[str, datetime] = {}

    for sym in syms:
        df_l = data[(sym.name, cluster["low"])]
        df_m = data[(sym.name, cluster["med"])]
        df_h = data[(sym.name, cluster["high"])]

        df, longs, shorts = run_cluster(
            df=df_l,
            df_m=df_m,
            df_h=df_h,
            cluster=cluster,
            show_length=1 if latest_only else n_bars,
        )
        if df.empty:
            continue

        row_indexes = [len(df) - 1] if latest_only else range(len(df))
        latest_bars[sym.name] = df.index[-1].to_pydatetime()

        for i in row_indexes:
            ts = df.index[i].to_pydatetime()
            price = float(df["close"].iloc[i])
            if longs[i]:
                signals.append(
                    Signal(sym.name, cluster_name, "long", ts, price, source)
                )
            if shorts[i]:
                signals.append(
                    Signal(sym.name, cluster_name, "short", ts, price, source)
                )

    return ScheduledRunResult(source, cluster_name, latest_bars, signals)


def persist_scheduled_signals(
    result: ScheduledRunResult, store: SignalStore
) -> list[Signal]:
    """Persist scheduled signals for a source+cluster run."""
    pending: list[Signal] = []
    for signal in result.signals:
        last_bar = store.get_last_bar_ts(
            signal.source, signal.symbol, signal.cluster
        )
        if last_bar is not None and last_bar >= signal.timestamp:
            continue
        store.insert_signal(
            StoredSignal(
                source=signal.source,
                symbol=signal.symbol,
                cluster=signal.cluster,
                direction=signal.direction,
                timestamp=signal.timestamp,
                price=signal.price,
            )
        )
        pending.append(signal)

    return pending


def mark_scheduled_run_processed(
    result: ScheduledRunResult, store: SignalStore
) -> None:
    """Advance source+cluster progress after downstream side effects succeed."""
    for symbol, latest_bar in result.latest_bars.items():
        store.set_last_bar_ts(result.source, symbol, result.cluster, latest_bar)
