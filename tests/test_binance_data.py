"""Tests for binance_data.py — targeting 100% coverage."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import binance_data
from binance_data import (
    _cache_path,
    _read_cache,
    _to_binance_interval,
    _write_cache,
    fetch_klines,
    fetch_klines_full,
    fetch_multi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW_MS = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
_BASE_MS = _NOW_MS - 3_600_000  # 1h ago


def _make_kline(open_time_ms: int, o=100, h=105, l=95, c=102, v=10):
    close_time_ms = open_time_ms + 300_000 - 1
    return [
        open_time_ms, str(o), str(h), str(l), str(c), str(v),
        close_time_ms, "1000", "50", "5", "500", "0",
    ]


def _mock_urlopen(klines):
    body = json.dumps(klines).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# _to_binance_interval
# ---------------------------------------------------------------------------


def test_to_binance_interval_known():
    assert _to_binance_interval("5min") == "5m"
    assert _to_binance_interval("1h") == "1h"
    assert _to_binance_interval("4h") == "4h"
    assert _to_binance_interval("1D") == "1d"
    assert _to_binance_interval("1W") == "1w"
    assert _to_binance_interval("1ME") == "1M"


def test_to_binance_interval_unsupported():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        _to_binance_interval("2min")


# ---------------------------------------------------------------------------
# fetch_klines
# ---------------------------------------------------------------------------


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_basic(mock_urlopen):
    klines = [_make_kline(_BASE_MS + i * 300_000) for i in range(5)]
    mock_urlopen.return_value = _mock_urlopen(klines)
    df = fetch_klines("BTCUSDT", "5min", limit=5)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) <= 5


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_futures_usdt(mock_urlopen):
    mock_urlopen.return_value = _mock_urlopen([_make_kline(_BASE_MS)])
    fetch_klines("BTCUSDT", "5min", market="futures-usdt")
    url_called = mock_urlopen.call_args[0][0]
    assert "fapi.binance.com" in url_called


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_futures_coin(mock_urlopen):
    mock_urlopen.return_value = _mock_urlopen([_make_kline(_BASE_MS)])
    fetch_klines("BTCUSD_PERP", "5min", market="futures-coin")
    url_called = mock_urlopen.call_args[0][0]
    assert "dapi.binance.com" in url_called


def test_fetch_klines_unknown_market():
    with pytest.raises(ValueError, match="Unknown market"):
        fetch_klines(market="invalid")


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_start_and_end_time(mock_urlopen):
    mock_urlopen.return_value = _mock_urlopen([_make_kline(_BASE_MS)])
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)
    fetch_klines("BTCUSDT", "5min", start_time=start, end_time=end)
    url_called = mock_urlopen.call_args[0][0]
    assert "startTime=" in url_called
    assert "endTime=" in url_called


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_closed_only_false(mock_urlopen):
    future_kline = _make_kline(_BASE_MS)
    future_kline[6] = _NOW_MS + 600_000
    mock_urlopen.return_value = _mock_urlopen([future_kline])
    df = fetch_klines("BTCUSDT", "5min", closed_only=False)
    assert len(df) == 1


@patch("binance_data.urllib.request.urlopen")
def test_fetch_klines_closed_only_drops_open_bar(mock_urlopen):
    future_kline = _make_kline(_BASE_MS)
    future_kline[6] = _NOW_MS + 600_000
    mock_urlopen.return_value = _mock_urlopen([future_kline])
    df = fetch_klines("BTCUSDT", "5min", closed_only=True)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# fetch_klines_full
# ---------------------------------------------------------------------------


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_single_page(mock_fk):
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [10, 11, 12, 13, 14],
        },
        index=idx,
    )
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=5, use_cache=False)
    assert len(result) == 5


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_pagination(mock_fk):
    def make_chunk(start, n, base_price):
        idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
        return pd.DataFrame(
            {
                "open": range(base_price, base_price + n),
                "high": range(base_price + 5, base_price + 5 + n),
                "low": range(base_price - 5, base_price - 5 + n),
                "close": range(base_price + 1, base_price + 1 + n),
                "volume": range(10, 10 + n),
            },
            index=idx,
        )

    # n_bars=2000 → first call limit=1500, returns 1500 (triggers pagination),
    # second call returns 500 (< 500, stops)
    mock_fk.side_effect = [
        make_chunk("2024-01-02 01:00", 1500, 200),
        make_chunk("2024-01-01 00:00", 300, 100),
    ]
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=2000, use_cache=False)
    assert mock_fk.call_count == 2
    assert len(result) == 1800


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_empty(mock_fk):
    mock_fk.return_value = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=100, use_cache=False)
    assert result.empty


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_partial_page_stops(mock_fk):
    """When a chunk returns fewer rows than requested, pagination stops."""
    idx = pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
        index=idx,
    )
    fetch_klines_full("BTCUSDT", "5min", n_bars=499, use_cache=False)
    assert mock_fk.call_count == 1


# ---------------------------------------------------------------------------
# fetch_multi
# ---------------------------------------------------------------------------


@patch("binance_data.fetch_klines_full")
def test_fetch_multi_returns_keyed_dict(mock_fkf):
    """fetch_multi returns dict keyed by (symbol, interval) with correct data."""
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    dummy = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
        index=idx,
    )
    mock_fkf.return_value = dummy

    tasks = [("BTCUSDT", "5min"), ("ETHUSDT", "4h")]
    result = fetch_multi(tasks, n_bars=100, market="futures-usdt", max_workers=2)

    assert set(result.keys()) == {("BTCUSDT", "5min"), ("ETHUSDT", "4h")}
    assert len(result[("BTCUSDT", "5min")]) == 10
    assert mock_fkf.call_count == 2


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_df(start="2024-01-01", periods=100, freq="5min"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(5, 5 + periods),
            "low": range(-5, -5 + periods),
            "close": range(1, 1 + periods),
            "volume": [10] * periods,
        },
        index=idx,
    )


def test_cache_path():
    p = _cache_path("BTCUSDT", "5min", "futures-usdt")
    assert p.parts[-3:] == ("futures-usdt", "BTCUSDT", "5min.parquet")


def test_read_cache_missing(tmp_path):
    assert _read_cache(tmp_path / "nope.parquet") is None


def test_read_cache_corrupt(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    assert _read_cache(bad) is None
    assert not bad.exists()


def test_read_cache_wrong_schema(tmp_path):
    wrong = tmp_path / "wrong.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(wrong, engine="fastparquet")
    assert _read_cache(wrong) is None
    assert not wrong.exists()


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "test.parquet"
    df = _make_df(periods=10)
    _write_cache(path, df)
    loaded = _read_cache(path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_write_cache_atomic_on_error(tmp_path):
    """If to_parquet raises, no partial file is left behind."""
    path = tmp_path / "sub" / "test.parquet"
    with patch("binance_data.pd.DataFrame.to_parquet", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _write_cache(path, _make_df(periods=5))
    # No .tmp file left behind, and target doesn't exist
    assert not path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# Cache integration with fetch_klines_full
# ---------------------------------------------------------------------------


@patch("binance_data.fetch_klines")
def test_cache_populated_on_first_run(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(binance_data, "_CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
        index=idx,
    )
    fetch_klines_full("BTCUSDT", "5min", n_bars=50, market="spot", use_cache=True)
    cache_file = tmp_path / "spot" / "BTCUSDT" / "5min.parquet"
    assert cache_file.exists()


@patch("binance_data.fetch_klines")
def test_cache_tail_only_on_second_run(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(binance_data, "_CACHE_DIR", tmp_path)

    # Pre-populate cache with 100 bars
    cached = _make_df("2024-01-01", periods=100, freq="5min")
    cache_file = tmp_path / "spot" / "BTCUSDT" / "5min.parquet"
    _write_cache(cache_file, cached)

    # Tail fetch returns 5 new bars
    tail_idx = pd.date_range("2024-01-01 08:10", periods=5, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 200, "high": 210, "low": 190, "close": 205, "volume": 20},
        index=tail_idx,
    )

    result = fetch_klines_full(
        "BTCUSDT", "5min", n_bars=200, market="spot",
        closed_only=False, use_cache=True,
    )
    # Should have called fetch_klines with start_time (tail fetch), not a full pagination
    call_kwargs = mock_fk.call_args
    assert call_kwargs.kwargs.get("start_time") or call_kwargs[1].get("start_time") \
        or (len(call_kwargs[0]) > 2 and call_kwargs[0][2] is not None)
    assert len(result) > len(cached) - 1  # got cached + tail minus dedup


@patch("binance_data.fetch_klines")
def test_use_cache_false_skips_cache(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(binance_data, "_CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
        index=idx,
    )
    fetch_klines_full("BTCUSDT", "5min", n_bars=10, market="spot", use_cache=False)
    cache_file = tmp_path / "spot" / "BTCUSDT" / "5min.parquet"
    assert not cache_file.exists()
