import pandas as pd

import main
from universe import Symbol


def test_generate_signals_uses_requested_output_window(monkeypatch):
    symbol = Symbol("TEST", "binance", "futures-usdt")
    fetch_sizes: list[int] = []
    show_lengths: list[int | None] = []

    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)

    def fake_fetch_all(universe, n_bars):
        fetch_sizes.append(n_bars)
        return {
            (symbol.name, tf): df
            for cluster in main.ALL_CLUSTERS.values()
            for tf in cluster.values()
        }

    def fake_run_cluster(
        *, df, df_m, df_h, cluster, show_length, pivot_length=main.PIVOT_LENGTH
    ):
        show_lengths.append(show_length)
        longs = pd.Series([False, True, False]).to_numpy(dtype=bool)
        shorts = pd.Series([False, False, False]).to_numpy(dtype=bool)
        return df, longs, shorts

    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(main, "run_cluster", fake_run_cluster)

    signals = main.generate_signals(universe=[symbol], n_bars=123)

    assert fetch_sizes == [123 + main.WARMUP_BARS]
    assert show_lengths == [123] * len(main.ALL_CLUSTERS)
    assert len(signals) == len(main.ALL_CLUSTERS)


def test_generate_signals_honors_explicit_show_length(monkeypatch):
    symbol = Symbol("TEST", "binance", "futures-usdt")
    show_lengths: list[int | None] = []

    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=idx)

    def fake_fetch_all(universe, n_bars):
        return {
            (symbol.name, tf): df
            for cluster in main.ALL_CLUSTERS.values()
            for tf in cluster.values()
        }

    def fake_run_cluster(
        *, df, df_m, df_h, cluster, show_length, pivot_length=main.PIVOT_LENGTH
    ):
        show_lengths.append(show_length)
        longs = pd.Series([False, False]).to_numpy(dtype=bool)
        shorts = pd.Series([False, True]).to_numpy(dtype=bool)
        return df, longs, shorts

    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(main, "run_cluster", fake_run_cluster)

    signals = main.generate_signals(universe=[symbol], n_bars=123, show_length=7)

    assert show_lengths == [7] * len(main.ALL_CLUSTERS)
    assert len(signals) == len(main.ALL_CLUSTERS)
