"""Fetch OHLCV kline data from Binance public REST API."""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

# ---------------------------------------------------------------------------
# Local parquet cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "klines"
_EXPECTED_COLS = {"open", "high", "low", "close", "volume"}


def _cache_path(symbol: str, interval: str, market: str) -> Path:
    return _CACHE_DIR / market / symbol / f"{interval}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, engine="fastparquet")
        if not _EXPECTED_COLS.issubset(df.columns):
            path.unlink()
            return None
        return df
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path, engine="fastparquet")
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _fetch_tail(
    symbol: str, interval: str, market: str, start_time: datetime
) -> pd.DataFrame:
    """Fetch bars from *start_time* to now, paginating forward."""
    _CHUNK = 499
    chunks: list[pd.DataFrame] = []
    cursor = start_time

    while True:
        chunk = fetch_klines(
            symbol, interval, limit=_CHUNK,
            start_time=cursor, market=market, closed_only=False,
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        if len(chunk) < _CHUNK:
            break
        cursor = chunk.index[-1].to_pydatetime()

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(chunks).sort_index().pipe(
        lambda d: d[~d.index.duplicated(keep="last")]
    )

_SPOT_URL = "https://api.binance.com/api/v3/klines"
_FUTURES_USDT_URL = "https://fapi.binance.com/fapi/v1/klines"
_FUTURES_COIN_URL = "https://dapi.binance.com/dapi/v1/klines"

# Map pandas resample/offset strings → Binance interval strings
_TF_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1D": "1d",
    "3D": "3d",
    "1W": "1w",
    "1ME": "1M",
}


def _to_binance_interval(tf: str) -> str:
    if tf in _TF_MAP:
        return _TF_MAP[tf]
    raise ValueError(f"Unsupported timeframe '{tf}'. Supported: {list(_TF_MAP)}")


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5min",
    limit: int = 1000,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    market: str = "futures-usdt",
    closed_only: bool = True,
) -> pd.DataFrame:
    """Fetch klines from Binance and return an OHLCV DataFrame.

    Parameters
    ----------
    symbol:
        Binance trading pair, e.g. ``"BTCUSDT"`` (spot/futures-usdt) or
        ``"BTCUSD_PERP"`` (futures-coin).
    interval:
        Timeframe in pandas notation (``"5min"``, ``"1h"``, ``"4h"``…).
    limit:
        Number of bars to fetch (max 1500 per request).
    start_time:
        Fetch bars starting from this UTC datetime (inclusive).
    end_time:
        Fetch bars up to this UTC datetime (inclusive).
    market:
        ``"spot"`` (default), ``"futures-usdt"`` (USDT-margined perpetuals),
        or ``"futures-coin"`` (coin-margined perpetuals, e.g. BTCUSD_PERP).
    closed_only:
        If ``True`` (default), drop the current still-open bar from the result.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: UTC DatetimeIndex named ``"timestamp"``.
    """
    _market_urls = {
        "spot": _SPOT_URL,
        "futures-usdt": _FUTURES_USDT_URL,
        "futures-coin": _FUTURES_COIN_URL,
    }
    if market not in _market_urls:
        raise ValueError(
            f"Unknown market '{market}'. Choose from: {list(_market_urls)}"
        )
    base_url = _market_urls[market]

    binance_interval = _to_binance_interval(interval)

    params = f"symbol={symbol}&interval={binance_interval}&limit={limit}"
    if start_time is not None:
        ms = int(start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
        params += f"&startTime={ms}"
    if end_time is not None:
        ms = int(end_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
        params += f"&endTime={ms}"

    url = f"{base_url}?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        raw = json.loads(resp.read())

    # Each element: [open_time, open, high, low, close, volume, ...]
    df = pd.DataFrame(
        raw,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")[
        ["open", "high", "low", "close", "volume", "close_time"]
    ]
    df = df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )

    if closed_only:
        now = datetime.now(tz=timezone.utc)
        df = df[df["close_time"] < now]

    result = df.drop(columns="close_time")
    result.attrs["symbol"] = symbol
    result.attrs["timeframe"] = interval
    return result


def _fetch_full_no_cache(
    symbol: str, interval: str, n_bars: int, market: str
) -> pd.DataFrame:
    """Backward-paginating fetch (no caching)."""
    # Binance rate-limit weight doubles at limit>=500 (2→5) and again at
    # >=1000 (5→10).  Fetching 499 per request (weight 2) uses ~45% less
    # weight budget than 1500 (weight 10) for the same total bars.
    _CHUNK = 499
    chunks: list[pd.DataFrame] = []
    end_time: datetime | None = None
    remaining = n_bars

    while remaining > 0:
        limit = min(remaining, _CHUNK)
        chunk = fetch_klines(
            symbol,
            interval,
            limit=limit,
            end_time=end_time,
            market=market,
            closed_only=False,
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        if len(chunk) < limit:
            break
        oldest_ms = int(chunk.index[0].timestamp() * 1000) - 1
        end_time = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(reversed(chunks)).sort_index().drop_duplicates()


def fetch_klines_full(
    symbol: str = "BTCUSDT",
    interval: str = "5min",
    n_bars: int = 5000,
    market: str = "spot",
    closed_only: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch more than 1000 bars by paginating Binance kline requests.

    When *use_cache* is ``True`` (default), bars are cached locally as
    Parquet files.  Subsequent calls only fetch the tail (new bars since
    the last cached timestamp), dramatically reducing API weight.

    Parameters
    ----------
    symbol:
        Binance trading pair.
    interval:
        Timeframe in pandas notation.
    n_bars:
        Total number of bars to retrieve.
    market:
        ``"spot"``, ``"futures-usdt"``, or ``"futures-coin"``.
    closed_only:
        If ``True`` (default), drop the current still-open bar from the result.
    use_cache:
        If ``True`` (default), read/write a local Parquet cache and only
        fetch new bars from Binance.
    """
    cached = _read_cache(_cache_path(symbol, interval, market)) if use_cache else None

    if cached is not None and not cached.empty:
        # Drop last cached row — it may have been an incomplete candle.
        cached = cached.iloc[:-1]
        if not cached.empty:
            fetch_start = cached.index[-1].to_pydatetime()
            tail = _fetch_tail(symbol, interval, market, start_time=fetch_start)
            result = pd.concat([cached, tail]).sort_index()
            result = result[~result.index.duplicated(keep="last")]
        else:
            result = _fetch_full_no_cache(symbol, interval, n_bars, market)
    else:
        result = _fetch_full_no_cache(symbol, interval, n_bars, market)

    # Trim to requested size
    result = result.iloc[-n_bars:]

    # Persist before applying closed_only filter
    if use_cache and not result.empty:
        _write_cache(_cache_path(symbol, interval, market), result)

    if closed_only and not result.empty:
        now = datetime.now(tz=timezone.utc)
        last_open = result.index[-1]
        expected_close = last_open + pd.tseries.frequencies.to_offset(interval)
        if expected_close > now:
            result = result.iloc[:-1]

    result.attrs["symbol"] = symbol
    result.attrs["timeframe"] = interval
    return result


def fetch_multi(
    tasks: list[tuple[str, str]],
    n_bars: int = 2000,
    market: str = "futures-usdt",
    max_workers: int = 6,
    closed_only: bool = True,
    use_cache: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch klines for multiple (symbol, interval) pairs in parallel.

    Parameters
    ----------
    tasks:
        List of ``(symbol, interval)`` tuples, e.g.
        ``[("BTCUSDT", "5min"), ("ETHUSDT", "4h")]``.
    n_bars:
        Number of bars to fetch per task.
    market:
        ``"spot"``, ``"futures-usdt"``, or ``"futures-coin"``.
    max_workers:
        Maximum concurrent requests (default 6, conservative vs Binance
        1200 weight/min limit).
    closed_only:
        If ``True`` (default), drop still-open bars.
    use_cache:
        If ``True`` (default), use local Parquet cache per task.

    Returns
    -------
    dict[tuple[str, str], pd.DataFrame]
        Keyed by ``(symbol, interval)``.
    """

    def _fetch(task: tuple[str, str]) -> tuple[tuple[str, str], pd.DataFrame]:
        symbol, interval = task
        df = fetch_klines_full(
            symbol, interval, n_bars=n_bars, market=market,
            closed_only=closed_only, use_cache=use_cache,
        )
        return task, df

    results: dict[tuple[str, str], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, df in pool.map(_fetch, tasks):
            results[key] = df
    return results


if __name__ == "__main__":
    # USDT-margined perpetual (most liquid BTC perp on Binance)
    df = fetch_klines("BTCUSDT", "1W", limit=100, market="futures-usdt")
    print(df.tail())
    print(f"\n{len(df)} bars fetched, index: {df.index[0]} → {df.index[-1]}")
