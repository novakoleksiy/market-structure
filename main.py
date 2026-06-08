"""Entry point: multi-source signal generator across Binance & OANDA."""

import argparse
from pathlib import Path

import signal_engine
import telegram_notifier
from clusters import ALL_CLUSTERS
from signal_engine import Signal, generate_signals
from signal_service import (
    generate_source_cluster_signals,
    mark_scheduled_run_processed,
    persist_scheduled_signals,
)
from signal_store import SignalStore
from universe import Symbol

# -- Config -----------------------------------------------------------------
DEFAULT_DB_PATH = Path(__file__).resolve().parent / ".cache" / "signals.sqlite3"
PIVOT_LENGTH = signal_engine.PIVOT_LENGTH
WARMUP_BARS = signal_engine.WARMUP_BARS
fetch_all = signal_engine.fetch_all
run_cluster = signal_engine.run_cluster


def run_scheduled_cluster(
    source: str,
    cluster_name: str,
    universe: list[Symbol] | None = None,
    n_bars: int = 2000,
    db_path: str | Path = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> list[Signal]:
    """Run one source-aware scheduled cluster job and persist new signals."""
    result = generate_source_cluster_signals(
        source,
        cluster_name,
        universe=universe,
        n_bars=n_bars,
        latest_only=False,
        use_cache=use_cache,
    )
    store = SignalStore(db_path)
    signals = persist_scheduled_signals(result, store)
    telegram_notifier.notify_new_signals(signals, source, cluster_name)
    mark_scheduled_run_processed(result, store)
    return signals


def latest_signals_per_combo(signals: list[Signal]) -> list[Signal]:
    """Return the newest signal for each source+symbol+cluster combination."""
    latest: dict[tuple[str, str, str], Signal] = {}
    for signal in signals:
        key = (signal.source, signal.symbol, signal.cluster)
        current = latest.get(key)
        if current is None or signal.timestamp > current.timestamp:
            latest[key] = signal
    return sorted(
        latest.values(), key=lambda sig: (sig.source, sig.symbol, sig.cluster)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run_source_cluster = subparsers.add_parser(
        "run-source-cluster",
        help="Run one scheduled source+cluster job",
    )
    run_source_cluster.add_argument(
        "--source", required=True, choices=["binance", "oanda"]
    )
    run_source_cluster.add_argument(
        "--cluster", required=True, choices=sorted(ALL_CLUSTERS)
    )
    run_source_cluster.add_argument("--bars", type=int, default=2000)
    run_source_cluster.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    run_source_cluster.add_argument(
        "--no-cache", action="store_true", help="Disable parquet market-data cache"
    )

    bt = subparsers.add_parser(
        "backtest",
        help="Backtest cluster signals with ATR-based SL/TP across the universe",
    )
    bt.add_argument(
        "--clusters",
        default=",".join(sorted(ALL_CLUSTERS)),
        help="Comma-separated clusters to run (default: all)",
    )
    bt.add_argument("--bars", type=int, default=3000)
    bt.add_argument(
        "--source",
        choices=["binance", "oanda"],
        help="Restrict the universe to one data source",
    )
    bt.add_argument("--atr-period", type=int, default=14)
    bt.add_argument("--sl-mult", type=float, default=1.0)
    bt.add_argument("--tp-mult", type=float, default=2.0)
    bt.add_argument(
        "--no-opposite-exit",
        action="store_true",
        help="Do not exit early on an opposite-direction signal",
    )
    bt.add_argument(
        "--interactive",
        action="store_true",
        help="Animated lightweight-charts playback (single symbol/cluster)",
    )
    bt.add_argument("--symbol", default="BTCUSDT", help="Symbol for --interactive")
    bt.add_argument("--cluster", default="C1", help="Cluster for --interactive")
    bt.add_argument("--speed", type=float, default=20.0, help="Bars/sec for --interactive")
    bt.add_argument(
        "--final",
        action="store_true",
        help="Render the completed backtest at once instead of candle-by-candle playback",
    )

    return parser


def _parse_clusters(raw: str) -> list[str]:
    """Validate and return cluster names from a comma-separated string."""
    names = [c.strip().upper() for c in raw.split(",") if c.strip()]
    unknown = [c for c in names if c not in ALL_CLUSTERS]
    if unknown:
        raise SystemExit(
            f"Unknown cluster(s): {unknown}. Choose from {sorted(ALL_CLUSTERS)}"
        )
    return names


def run_backtest_command(args: argparse.Namespace) -> int:
    """Dispatch the `backtest` subcommand (headless report or interactive GUI)."""
    if args.interactive:
        import backtest_viz

        backtest_viz.play_backtest(
            symbol=args.symbol,
            cluster_name=args.cluster.upper(),
            n_bars=args.bars,
            speed=args.speed,
            atr_period=args.atr_period,
            sl_mult=args.sl_mult,
            tp_mult=args.tp_mult,
            opposite_exit=not args.no_opposite_exit,
            final=args.final,
        )
        return 0

    import backtest
    from universe import build_universe

    universe = None
    if args.source:
        universe = [s for s in build_universe() if s.source == args.source]

    result = backtest.run_backtest(
        universe=universe,
        clusters=_parse_clusters(args.clusters),
        n_bars=args.bars,
        atr_period=args.atr_period,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        opposite_exit=not args.no_opposite_exit,
    )
    backtest.print_report(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-source-cluster":
        signals = run_scheduled_cluster(
            source=args.source,
            cluster_name=args.cluster,
            n_bars=args.bars,
            db_path=args.db_path,
            use_cache=not args.no_cache,
        )
        for sig in signals:
            print(
                f"[{sig.source}:{sig.cluster}] {sig.direction.upper():5s} {sig.symbol:12s} "
                f"@ {sig.price:.5f}  ({sig.timestamp})"
            )
        print(f"\n{len(signals)} new signals persisted")
        return 0

    if args.command == "backtest":
        return run_backtest_command(args)

    signals = latest_signals_per_combo(generate_signals(n_bars=3000))
    for sig in signals:
        print(
            f"[{sig.source}:{sig.cluster}] {sig.direction.upper():5s} {sig.symbol:12s} "
            f"@ {sig.price:.5f}  ({sig.timestamp})"
        )
    print(f"\n{len(signals)} latest signals total")
    return 0


# -- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
