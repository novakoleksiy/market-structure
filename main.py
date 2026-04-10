"""Entry point: multi-source signal generator across Binance & OANDA."""

import argparse
from pathlib import Path

import signal_engine
from clusters import ALL_CLUSTERS
from signal_engine import Signal, generate_signals
from signal_service import generate_source_cluster_signals, persist_scheduled_signals
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
        latest_only=True,
        use_cache=use_cache,
    )
    store = SignalStore(db_path)
    return persist_scheduled_signals(result, store)


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

    return parser


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

    signals = generate_signals()
    for sig in signals:
        print(
            f"[{sig.cluster}] {sig.direction.upper():5s} {sig.symbol:12s} "
            f"@ {sig.price:.5f}  ({sig.timestamp})"
        )
    print(f"\n{len(signals)} signals total")
    return 0


# -- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
