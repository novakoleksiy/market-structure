"""Source-aware signal generation for scheduled runs."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import binance_data
import tradfi_data
from clusters import get_cluster
from signal_engine import (
    WARMUP_BARS,
    Signal,
    cluster_frames,
    run_cluster,
    signals_from_cluster_output,
)
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


def _cluster_tasks(
    syms: list[Symbol], cluster: dict[str, str]
) -> list[tuple[str, str]]:
    """Return the exact symbol/timeframe requests needed for one cluster."""
    return [(sym.name, tf) for sym in syms for tf in cluster.values()]


def _fetch_multi_for_source(
    source: str,
    tasks: list[tuple[str, str]],
    n_bars: int,
    source_param: str,
    use_cache: bool,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch one provider/source-param group with the provider-specific client."""
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


def _fetch_source_param_group(
    source: str,
    syms: list[Symbol],
    cluster: dict[str, str],
    n_bars: int,
    use_cache: bool,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch candles for symbols that share the same provider market bucket."""
    source_param = syms[0].source_param
    return _fetch_multi_for_source(
        source,
        _cluster_tasks(syms, cluster),
        n_bars,
        source_param,
        use_cache,
    )


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

    data: dict[tuple[str, str], pd.DataFrame] = {}
    for source_param in sorted({sym.source_param for sym in syms}):
        group = [sym for sym in syms if sym.source_param == source_param]
        data.update(
            _fetch_source_param_group(source, group, cluster, n_bars, use_cache)
        )
    return data


def _run_scheduled_symbol(
    sym: Symbol,
    source: str,
    cluster_name: str,
    cluster: dict[str, str],
    data: dict[tuple[str, str], pd.DataFrame],
    n_bars: int,
    latest_only: bool,
) -> tuple[datetime | None, list[Signal]]:
    """Run one scheduled symbol and return its latest bar plus emitted signals."""
    df_l, df_m, df_h = cluster_frames(data, sym.name, cluster)
    df, longs, shorts = run_cluster(
        df=df_l,
        df_m=df_m,
        df_h=df_h,
        cluster=cluster,
        show_length=1 if latest_only else n_bars,
    )
    if df.empty:
        return None, []

    row_indexes = [len(df) - 1] if latest_only else range(len(df))
    signals = signals_from_cluster_output(
        sym.name,
        source,
        cluster_name,
        df,
        longs,
        shorts,
        row_indexes=row_indexes,
    )
    return df.index[-1].to_pydatetime(), signals


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
        latest_bar, symbol_signals = _run_scheduled_symbol(
            sym,
            source,
            cluster_name,
            cluster,
            data,
            n_bars,
            latest_only,
        )
        if latest_bar is None:
            continue
        latest_bars[sym.name] = latest_bar
        signals.extend(symbol_signals)

    return ScheduledRunResult(source, cluster_name, latest_bars, signals)


def persist_scheduled_signals(
    result: ScheduledRunResult, store: SignalStore
) -> list[Signal]:
    """Persist scheduled signals for a source+cluster run."""
    pending: list[Signal] = []
    for signal in result.signals:
        latest_bar = result.latest_bars.get(signal.symbol)
        last_bar = store.get_last_bar_ts(signal.source, signal.symbol, signal.cluster)
        if last_bar is None and signal.timestamp != latest_bar:
            continue
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
