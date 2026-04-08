"""Tests for tradfi_data.py — targeting 100% coverage."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import tradfi_data
from data_cache import ParquetCacheAdapter
from tradfi_data import (
    _cache_path,
    _to_oanda_granularity,
    fetch_klines,
    fetch_klines_full,
    fetch_multi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candle(
    time_str: str,
    o=1.09,
    h=1.10,
    low=1.08,
    c=1.095,
    v=100,
    complete=True,
):
    return {
        "time": time_str,
        "mid": {"o": str(o), "h": str(h), "l": str(low), "c": str(c)},
        "volume": v,
        "complete": complete,
    }


def _make_oanda_response(candles):
    body = json.dumps(
        {
            "instrument": "EUR_USD",
            "granularity": "M5",
            "candles": candles,
        }
    ).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# _to_oanda_granularity
# ---------------------------------------------------------------------------


def test_to_oanda_granularity_known():
    assert _to_oanda_granularity("5min") == "M5"
    assert _to_oanda_granularity("1h") == "H1"
    assert _to_oanda_granularity("4h") == "H4"
    assert _to_oanda_granularity("1D") == "D"
    assert _to_oanda_granularity("1W") == "W"
    assert _to_oanda_granularity("1ME") == "M"


def test_to_oanda_granularity_unsupported():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        _to_oanda_granularity("3min")


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OANDA_API", "test-key-123")
    from tradfi_data import _get_api_key

    assert _get_api_key() == "test-key-123"


def test_get_api_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OANDA_API", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OANDA_API=file-key-456\n")
    # Redirect the .env lookup to our tmp_path
    monkeypatch.setattr(
        tradfi_data,
        "_get_api_key",
        lambda: next(
            line.split("=", 1)[1].strip()
            for line in env_file.read_text().splitlines()
            if line.startswith("OANDA_API=")
        ),
    )
    assert tradfi_data._get_api_key() == "file-key-456"


# ---------------------------------------------------------------------------
# fetch_klines
# ---------------------------------------------------------------------------


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_basic(mock_urlopen, _mock_key):
    candles = [_make_candle(f"2024-01-15T0{i}:00:00.000000000Z") for i in range(5)]
    mock_urlopen.return_value = _make_oanda_response(candles)
    df = fetch_klines("EUR_USD", "5min", limit=5)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert df.attrs["symbol"] == "EUR_USD"
    assert df.attrs["timeframe"] == "5min"


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_auth_header(mock_urlopen, _mock_key):
    mock_urlopen.return_value = _make_oanda_response(
        [
            _make_candle("2024-01-15T00:00:00.000000000Z"),
        ]
    )
    fetch_klines("EUR_USD", "5min")
    req = mock_urlopen.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer test-key"


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_url_contains_instrument(mock_urlopen, _mock_key):
    mock_urlopen.return_value = _make_oanda_response(
        [
            _make_candle("2024-01-15T00:00:00.000000000Z"),
        ]
    )
    fetch_klines("SPX500_USD", "1D", asset_class="indices")
    req = mock_urlopen.call_args[0][0]
    assert "SPX500_USD" in req.full_url


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_start_and_end_time(mock_urlopen, _mock_key):
    mock_urlopen.return_value = _make_oanda_response(
        [
            _make_candle("2024-01-15T00:00:00.000000000Z"),
        ]
    )
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)
    fetch_klines("EUR_USD", "5min", start_time=start, end_time=end)
    req = mock_urlopen.call_args[0][0]
    assert "from=" in req.full_url
    assert "to=" in req.full_url


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_closed_only_false(mock_urlopen, _mock_key):
    candles = [
        _make_candle("2024-01-15T00:00:00.000000000Z", complete=False),
    ]
    mock_urlopen.return_value = _make_oanda_response(candles)
    df = fetch_klines("EUR_USD", "5min", closed_only=False)
    assert len(df) == 1


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_closed_only_drops_incomplete(mock_urlopen, _mock_key):
    candles = [
        _make_candle("2024-01-15T00:00:00.000000000Z", complete=False),
    ]
    mock_urlopen.return_value = _make_oanda_response(candles)
    df = fetch_klines("EUR_USD", "5min", closed_only=True)
    assert len(df) == 0


@patch("tradfi_data._get_api_key", return_value="test-key")
@patch("tradfi_data.urllib.request.urlopen")
def test_fetch_klines_empty_response(mock_urlopen, _mock_key):
    mock_urlopen.return_value = _make_oanda_response([])
    df = fetch_klines("EUR_USD", "5min")
    assert df.empty
    assert df.attrs["symbol"] == "EUR_USD"


# ---------------------------------------------------------------------------
# fetch_klines_full
# ---------------------------------------------------------------------------


@patch("tradfi_data.fetch_klines")
def test_fetch_klines_full_single_page(mock_fk):
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {
            "open": [1.09 + i * 0.001 for i in range(5)],
            "high": [1.10 + i * 0.001 for i in range(5)],
            "low": [1.08 + i * 0.001 for i in range(5)],
            "close": [1.095 + i * 0.001 for i in range(5)],
            "volume": range(100, 105),
        },
        index=idx,
    )
    result = fetch_klines_full("EUR_USD", "5min", n_bars=5, use_cache=False)
    assert len(result) == 5


@patch("tradfi_data.fetch_klines")
def test_fetch_klines_full_pagination(mock_fk):
    def make_chunk(start, n, base_price):
        idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
        return pd.DataFrame(
            {
                "open": [base_price + i * 0.0001 for i in range(n)],
                "high": [base_price + 0.01 + i * 0.0001 for i in range(n)],
                "low": [base_price - 0.01 + i * 0.0001 for i in range(n)],
                "close": [base_price + 0.005 + i * 0.0001 for i in range(n)],
                "volume": range(100, 100 + n),
            },
            index=idx,
        )

    mock_fk.side_effect = [
        make_chunk("2024-01-02 01:00", 5000, 1.09),
        make_chunk("2024-01-01 00:00", 300, 1.08),
    ]
    result = fetch_klines_full("EUR_USD", "5min", n_bars=6000, use_cache=False)
    assert mock_fk.call_count == 2
    assert len(result) == 5300


@patch("tradfi_data.fetch_klines")
def test_fetch_klines_full_empty(mock_fk):
    mock_fk.return_value = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    result = fetch_klines_full("EUR_USD", "5min", n_bars=100, use_cache=False)
    assert result.empty


@patch("tradfi_data.fetch_klines")
def test_fetch_klines_full_partial_page_stops(mock_fk):
    """When a chunk returns fewer rows than requested, pagination stops."""
    idx = pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 1.09, "high": 1.10, "low": 1.08, "close": 1.095, "volume": 100},
        index=idx,
    )
    fetch_klines_full("EUR_USD", "5min", n_bars=5000, use_cache=False)
    assert mock_fk.call_count == 1


# ---------------------------------------------------------------------------
# fetch_multi
# ---------------------------------------------------------------------------


@patch("tradfi_data.fetch_klines_full")
def test_fetch_multi_returns_keyed_dict(mock_fkf):
    """fetch_multi returns dict keyed by (symbol, interval) with correct data."""
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    dummy = pd.DataFrame(
        {"open": 1.09, "high": 1.10, "low": 1.08, "close": 1.095, "volume": 100},
        index=idx,
    )
    mock_fkf.return_value = dummy

    tasks = [("EUR_USD", "5min"), ("XAU_USD", "4h")]
    result = fetch_multi(tasks, n_bars=100, asset_class="forex", max_workers=2)

    assert set(result.keys()) == {("EUR_USD", "5min"), ("XAU_USD", "4h")}
    assert len(result[("EUR_USD", "5min")]) == 10
    assert mock_fkf.call_count == 2


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_df(start="2024-01-01", periods=100, freq="5min"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": [1.09 + i * 0.001 for i in range(periods)],
            "high": [1.10 + i * 0.001 for i in range(periods)],
            "low": [1.08 + i * 0.001 for i in range(periods)],
            "close": [1.095 + i * 0.001 for i in range(periods)],
            "volume": [100] * periods,
        },
        index=idx,
    )


def test_cache_path():
    p = _cache_path("EUR_USD", "5min", "forex")
    assert p.parts[-4:] == ("oanda", "forex", "EUR_USD", "5min.parquet")


def test_read_cache_missing(tmp_path):
    assert ParquetCacheAdapter(tmp_path).read("nope") is None


def test_read_cache_corrupt(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    assert ParquetCacheAdapter(tmp_path).read("bad") is None
    assert not bad.exists()


def test_read_cache_wrong_schema(tmp_path):
    wrong = tmp_path / "wrong.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(wrong, engine="fastparquet")
    assert ParquetCacheAdapter(tmp_path).read("wrong") is None
    assert not wrong.exists()


def test_write_and_read_roundtrip(tmp_path):
    df = _make_df(periods=10)
    cache = ParquetCacheAdapter(tmp_path)
    cache.write(df, "test")
    loaded = cache.read("test")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_write_cache_atomic_on_error(tmp_path):
    """If to_parquet raises, no partial file is left behind."""
    path = tmp_path / "sub" / "test.parquet"
    with patch("data_cache.pd.DataFrame.to_parquet", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            ParquetCacheAdapter(tmp_path).write_path(path, _make_df(periods=5))
    assert not path.exists()
    assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# Cache integration with fetch_klines_full
# ---------------------------------------------------------------------------


@patch("tradfi_data.fetch_klines")
def test_cache_populated_on_first_run(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(tradfi_data, "_CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 1.09, "high": 1.10, "low": 1.08, "close": 1.095, "volume": 100},
        index=idx,
    )
    fetch_klines_full("EUR_USD", "5min", n_bars=50, asset_class="forex", use_cache=True)
    cache_file = tmp_path / "oanda" / "forex" / "EUR_USD" / "5min.parquet"
    assert cache_file.exists()


@patch("tradfi_data.fetch_klines")
def test_cache_tail_only_on_second_run(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(tradfi_data, "_CACHE_DIR", tmp_path)

    # Pre-populate cache with 100 bars
    cached = _make_df("2024-01-01", periods=100, freq="5min")
    cache_file = tmp_path / "oanda" / "forex" / "EUR_USD" / "5min.parquet"
    ParquetCacheAdapter(tmp_path).write_path(cache_file, cached)

    # Tail fetch returns 5 new bars
    tail_idx = pd.date_range("2024-01-01 08:10", periods=5, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 1.20, "high": 1.21, "low": 1.19, "close": 1.205, "volume": 200},
        index=tail_idx,
    )

    result = fetch_klines_full(
        "EUR_USD",
        "5min",
        n_bars=80,
        asset_class="forex",
        closed_only=False,
        use_cache=True,
    )
    call_kwargs = mock_fk.call_args
    assert (
        call_kwargs.kwargs.get("start_time")
        or call_kwargs[1].get("start_time")
        or (len(call_kwargs[0]) > 2 and call_kwargs[0][2] is not None)
    )
    assert len(result) == 80
    assert result.index[-1] == tail_idx[-1]


@patch("tradfi_data.fetch_klines")
def test_use_cache_false_skips_cache(mock_fk, tmp_path, monkeypatch):
    monkeypatch.setattr(tradfi_data, "_CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    mock_fk.return_value = pd.DataFrame(
        {"open": 1.09, "high": 1.10, "low": 1.08, "close": 1.095, "volume": 100},
        index=idx,
    )
    fetch_klines_full(
        "EUR_USD", "5min", n_bars=10, asset_class="forex", use_cache=False
    )
    cache_file = tmp_path / "oanda" / "forex" / "EUR_USD" / "5min.parquet"
    assert not cache_file.exists()


@patch("tradfi_data._fetch_full_no_cache")
@patch("tradfi_data.fetch_klines")
def test_cache_backfills_when_request_exceeds_cached_window(
    mock_fk, mock_full, tmp_path, monkeypatch
):
    monkeypatch.setattr(tradfi_data, "_CACHE_DIR", tmp_path)

    cached = _make_df("2024-01-01", periods=100, freq="5min")
    cache_file = tmp_path / "oanda" / "forex" / "EUR_USD" / "5min.parquet"
    ParquetCacheAdapter(tmp_path).write_path(cache_file, cached)

    full = _make_df("2023-12-25", periods=300, freq="5min")
    mock_full.return_value = full

    result = fetch_klines_full(
        "EUR_USD",
        "5min",
        n_bars=300,
        asset_class="forex",
        closed_only=False,
        use_cache=True,
    )

    mock_full.assert_called_once_with("EUR_USD", "5min", 300, "forex")
    mock_fk.assert_not_called()
    assert len(result) == 300
    assert result.index[0] == full.index[0]
