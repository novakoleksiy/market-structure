"""Tests for binance_data.py — targeting 100% coverage."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from binance_data import _to_binance_interval, fetch_klines, fetch_klines_full

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
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=5)
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

    # n_bars=2000 → first call limit=1000, returns 1000 (triggers pagination),
    # second call returns 500 (< 1000, stops)
    mock_fk.side_effect = [
        make_chunk("2024-01-01 12:00", 1000, 200),
        make_chunk("2024-01-01 00:00", 500, 100),
    ]
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=2000)
    assert mock_fk.call_count == 2
    assert len(result) == 1500


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_empty(mock_fk):
    mock_fk.return_value = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    result = fetch_klines_full("BTCUSDT", "5min", n_bars=100)
    assert result.empty


@patch("binance_data.fetch_klines")
def test_fetch_klines_full_partial_page_stops(mock_fk):
    idx = pd.date_range("2024-01-01", periods=500, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
        index=idx,
    )
    fetch_klines_full("BTCUSDT", "5min", n_bars=1000)
    assert mock_fk.call_count == 1
