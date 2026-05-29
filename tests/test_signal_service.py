from pathlib import Path

import pandas as pd
import pytest

from signal_engine import Signal
from signal_service import (
    ScheduledRunResult,
    fetch_source_cluster_data,
    mark_scheduled_run_processed,
    persist_scheduled_signals,
)
from signal_store import SignalStore
from universe import Symbol


def test_fetch_source_cluster_data_only_requests_cluster_timeframes(monkeypatch):
    symbol = Symbol("BTCUSDT", "binance", "futures-usdt")
    calls: list[tuple[list[tuple[str, str]], int, str, bool]] = []

    def fake_fetch_multi(tasks, n_bars, market, use_cache=True, **kwargs):
        calls.append((tasks, n_bars, market, use_cache))
        idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            },
            index=idx,
        )
        return {task: df for task in tasks}

    monkeypatch.setattr("signal_service.binance_data.fetch_multi", fake_fetch_multi)

    data = fetch_source_cluster_data(
        "binance",
        "C2",
        universe=[symbol],
        n_bars=321,
        use_cache=False,
    )

    assert list(data) == [("BTCUSDT", "30min"), ("BTCUSDT", "4h"), ("BTCUSDT", "1D")]
    assert calls == [
        (
            [("BTCUSDT", "30min"), ("BTCUSDT", "4h"), ("BTCUSDT", "1D")],
            321,
            "futures-usdt",
            False,
        )
    ]


def test_persist_scheduled_signals_suppresses_duplicate_bar(tmp_path):
    store = SignalStore(Path(tmp_path) / "signals.sqlite3")
    bar_ts = pd.Timestamp("2024-01-01T04:00:00Z").to_pydatetime()
    signal = Signal("BTCUSDT", "C3", "long", bar_ts, 123.45, "binance")
    result = ScheduledRunResult("binance", "C3", {signal.symbol: bar_ts}, [signal])

    first = persist_scheduled_signals(result, store)
    mark_scheduled_run_processed(result, store)
    second = persist_scheduled_signals(result, store)

    assert first == [signal]
    assert second == []


def test_persist_scheduled_signals_retries_until_bar_processed(tmp_path):
    store = SignalStore(Path(tmp_path) / "signals.sqlite3")
    bar_ts = pd.Timestamp("2024-01-01T04:00:00Z").to_pydatetime()
    signal = Signal("BTCUSDT", "C3", "long", bar_ts, 123.45, "binance")
    result = ScheduledRunResult("binance", "C3", {signal.symbol: bar_ts}, [signal])

    first = persist_scheduled_signals(result, store)
    second = persist_scheduled_signals(result, store)
    mark_scheduled_run_processed(result, store)
    third = persist_scheduled_signals(result, store)

    assert first == [signal]
    assert second == [signal]
    assert third == []


def test_persist_scheduled_signals_tracks_progress_per_symbol(tmp_path):
    store = SignalStore(Path(tmp_path) / "signals.sqlite3")
    fast_bar = pd.Timestamp("2024-01-01T10:00:00Z").to_pydatetime()
    lag_previous_bar = pd.Timestamp("2024-01-01T09:30:00Z").to_pydatetime()
    lag_new_bar = pd.Timestamp("2024-01-01T09:35:00Z").to_pydatetime()
    signal = Signal("SPX500_USD", "C1", "long", lag_new_bar, 4800.0, "oanda")

    store.set_last_bar_ts("oanda", "EUR_USD", "C1", fast_bar)
    store.set_last_bar_ts("oanda", "SPX500_USD", "C1", lag_previous_bar)
    result = ScheduledRunResult(
        "oanda",
        "C1",
        {"EUR_USD": fast_bar, "SPX500_USD": lag_new_bar},
        [signal],
    )

    pending = persist_scheduled_signals(result, store)
    mark_scheduled_run_processed(result, store)
    duplicate = persist_scheduled_signals(result, store)

    assert pending == [signal]
    assert duplicate == []
    assert store.get_last_bar_ts("oanda", "EUR_USD", "C1") == fast_bar
    assert store.get_last_bar_ts("oanda", "SPX500_USD", "C1") == lag_new_bar


def test_run_scheduled_cluster_persists_only_new_latest_bar(monkeypatch, tmp_path):
    import main

    symbol = Symbol("EUR_USD", "oanda", "forex")
    latest_bar = pd.Timestamp("2024-01-02T00:00:00Z").to_pydatetime()
    captured: list[tuple[str, str, list[Symbol] | None, int, bool, bool]] = []

    def fake_generate_source_cluster_signals(
        source,
        cluster_name,
        universe=None,
        n_bars=2000,
        latest_only=True,
        use_cache=True,
    ):
        captured.append(
            (source, cluster_name, universe, n_bars, latest_only, use_cache)
        )
        return ScheduledRunResult(
            source,
            cluster_name,
            {symbol.name: latest_bar},
            [Signal(symbol.name, cluster_name, "short", latest_bar, 1.2345, source)],
        )

    monkeypatch.setattr(
        main, "generate_source_cluster_signals", fake_generate_source_cluster_signals
    )
    notify_calls: list[tuple[list[Signal], str, str]] = []

    def fake_notify_new_signals(signals, source, cluster):
        notify_calls.append((signals, source, cluster))

    monkeypatch.setattr(
        main.telegram_notifier, "notify_new_signals", fake_notify_new_signals
    )

    first = main.run_scheduled_cluster(
        "oanda",
        "C4",
        universe=[symbol],
        n_bars=99,
        db_path=Path(tmp_path) / "signals.sqlite3",
        use_cache=False,
    )
    second = main.run_scheduled_cluster(
        "oanda",
        "C4",
        universe=[symbol],
        n_bars=99,
        db_path=Path(tmp_path) / "signals.sqlite3",
        use_cache=False,
    )

    assert first[0].symbol == symbol.name
    assert second == []
    assert captured == [
        ("oanda", "C4", [symbol], 99, True, False),
        ("oanda", "C4", [symbol], 99, True, False),
    ]
    assert notify_calls == [([first[0]], "oanda", "C4"), ([], "oanda", "C4")]


def test_run_scheduled_cluster_retries_notification_before_advancing_state(
    monkeypatch, tmp_path
):
    import main

    symbol = Symbol("EUR_USD", "oanda", "forex")
    latest_bar = pd.Timestamp("2024-01-02T00:00:00Z").to_pydatetime()
    signal = Signal(symbol.name, "C4", "short", latest_bar, 1.2345, "oanda")

    def fake_generate_source_cluster_signals(
        source,
        cluster_name,
        universe=None,
        n_bars=2000,
        latest_only=True,
        use_cache=True,
    ):
        return ScheduledRunResult(source, cluster_name, {symbol.name: latest_bar}, [signal])

    monkeypatch.setattr(
        main, "generate_source_cluster_signals", fake_generate_source_cluster_signals
    )
    notify_calls: list[list[Signal]] = []

    def fake_notify_new_signals(signals, source, cluster):
        notify_calls.append(signals)
        if len(notify_calls) == 1:
            raise RuntimeError("telegram unavailable")

    monkeypatch.setattr(
        main.telegram_notifier, "notify_new_signals", fake_notify_new_signals
    )

    db_path = Path(tmp_path) / "signals.sqlite3"
    with pytest.raises(RuntimeError, match="telegram unavailable"):
        main.run_scheduled_cluster(
            "oanda",
            "C4",
            universe=[symbol],
            db_path=db_path,
        )

    retry = main.run_scheduled_cluster(
        "oanda",
        "C4",
        universe=[symbol],
        db_path=db_path,
    )
    duplicate = main.run_scheduled_cluster(
        "oanda",
        "C4",
        universe=[symbol],
        db_path=db_path,
    )

    assert retry == [signal]
    assert duplicate == []
    assert notify_calls == [[signal], [signal], []]
