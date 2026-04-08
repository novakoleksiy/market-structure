"""Fetch OHLCV candle data from OANDA v20 REST API."""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from chart import plot_market_structure

# ---------------------------------------------------------------------------
# OANDA API configuration
# ---------------------------------------------------------------------------

_PRACTICE_URL = "https://api-fxpractice.oanda.com"
_LIVE_URL = "https://api-fxtrade.oanda.com"

# ---------------------------------------------------------------------------
# Local parquet cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "klines" / "oanda"
_EXPECTED_COLS = {"open", "high", "low", "close", "volume"}


def _get_api_key() -> str:
    """Return OANDA API key from env or .env file."""
    key = os.environ.get("OANDA_API")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OANDA_API="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "OANDA_API not found. Set it as an environment variable or in .env"
    )


def _base_url() -> str:
    env = os.environ.get("OANDA_ENV", "practice").lower()
    return _LIVE_URL if env == "live" else _PRACTICE_URL


def _cache_path(symbol: str, interval: str, asset_class: str) -> Path:
    return _CACHE_DIR / asset_class / symbol / f"{interval}.parquet"


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
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path, engine="fastparquet")
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Interval mapping
# ---------------------------------------------------------------------------

# Map pandas resample/offset strings → OANDA granularity strings
_TF_MAP = {
    "1min": "M1",
    "2min": "M2",
    "4min": "M4",
    "5min": "M5",
    "10min": "M10",
    "15min": "M15",
    "30min": "M30",
    "1h": "H1",
    "2h": "H2",
    "3h": "H3",
    "4h": "H4",
    "6h": "H6",
    "8h": "H8",
    "12h": "H12",
    "1D": "D",
    "1W": "W",
    "1ME": "M",
}


def _to_oanda_granularity(tf: str) -> str:
    if tf in _TF_MAP:
        return _TF_MAP[tf]
    raise ValueError(f"Unsupported timeframe '{tf}'. Supported: {list(_TF_MAP)}")


def _to_rfc3339(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------


def fetch_klines(
    symbol: str = "EUR_USD",
    interval: str = "5min",
    limit: int = 500,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    asset_class: str = "forex",
    closed_only: bool = True,
) -> pd.DataFrame:
    """Fetch candles from OANDA and return an OHLCV DataFrame.

    Parameters
    ----------
    symbol:
        OANDA instrument name, e.g. ``"EUR_USD"``, ``"SPX500_USD"``,
        ``"XAU_USD"``.
    interval:
        Timeframe in pandas notation (``"5min"``, ``"1h"``, ``"4h"``…).
    limit:
        Number of bars to fetch (max 5000 per request).
    start_time:
        Fetch bars starting from this UTC datetime (inclusive).
    end_time:
        Fetch bars up to this UTC datetime (inclusive).
    asset_class:
        ``"forex"``, ``"indices"``, or ``"commodities"``.
        Used only for cache directory organisation.
    closed_only:
        If ``True`` (default), drop the current still-open bar from the result.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: UTC DatetimeIndex named ``"timestamp"``.
    """
    granularity = _to_oanda_granularity(interval)

    url = (
        f"{_base_url()}/v3/instruments/{symbol}/candles"
        f"?granularity={granularity}&price=M&count={limit}"
    )
    if start_time is not None:
        url += f"&from={_to_rfc3339(start_time)}"
        if end_time is None:
            # OANDA: count + from → fetches `count` bars forward from `from`
            pass
        else:
            # OANDA: from + to → no count allowed
            url = (
                f"{_base_url()}/v3/instruments/{symbol}/candles"
                f"?granularity={granularity}&price=M"
                f"&from={_to_rfc3339(start_time)}&to={_to_rfc3339(end_time)}"
            )
    elif end_time is not None:
        url = (
            f"{_base_url()}/v3/instruments/{symbol}/candles"
            f"?granularity={granularity}&price=M&count={limit}"
            f"&to={_to_rfc3339(end_time)}"
        )

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {_get_api_key()}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    candles = data.get("candles", [])
    if not candles:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.attrs["symbol"] = symbol
        empty.attrs["timeframe"] = interval
        return empty

    rows = []
    for c in candles:
        mid = c["mid"]
        rows.append(
            {
                "timestamp": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
                "complete": c["complete"],
            }
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if closed_only:
        df = df[df["complete"]]

    result = df[["open", "high", "low", "close", "volume"]].copy()
    result.attrs["symbol"] = symbol
    result.attrs["timeframe"] = interval
    return result


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------


def _fetch_tail(
    symbol: str, interval: str, asset_class: str, start_time: datetime
) -> pd.DataFrame:
    """Fetch bars from *start_time* to now, paginating forward."""
    _CHUNK = 5000
    chunks: list[pd.DataFrame] = []
    cursor = start_time

    while True:
        chunk = fetch_klines(
            symbol,
            interval,
            limit=_CHUNK,
            start_time=cursor,
            asset_class=asset_class,
            closed_only=False,
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        if len(chunk) < _CHUNK:
            break
        cursor = chunk.index[-1].to_pydatetime()

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return (
        pd.concat(chunks)
        .sort_index()
        .pipe(lambda d: d[~d.index.duplicated(keep="last")])
    )


def _fetch_full_no_cache(
    symbol: str, interval: str, n_bars: int, asset_class: str
) -> pd.DataFrame:
    """Backward-paginating fetch (no caching)."""
    _CHUNK = 5000
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
            asset_class=asset_class,
            closed_only=False,
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        if len(chunk) < limit:
            break
        oldest_ts = chunk.index[0].to_pydatetime()
        end_time = oldest_ts

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(reversed(chunks)).sort_index().drop_duplicates()


# ---------------------------------------------------------------------------
# Public: full fetch with caching
# ---------------------------------------------------------------------------


def fetch_klines_full(
    symbol: str = "EUR_USD",
    interval: str = "5min",
    n_bars: int = 5000,
    asset_class: str = "forex",
    closed_only: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch more than 5000 bars by paginating OANDA candle requests.

    When *use_cache* is ``True`` (default), bars are cached locally as
    Parquet files.  Subsequent calls only fetch the tail (new bars since
    the last cached timestamp), dramatically reducing API calls.

    Parameters
    ----------
    symbol:
        OANDA instrument name.
    interval:
        Timeframe in pandas notation.
    n_bars:
        Total number of bars to retrieve.
    asset_class:
        ``"forex"``, ``"indices"``, or ``"commodities"``.
    closed_only:
        If ``True`` (default), drop the current still-open bar from the result.
    use_cache:
        If ``True`` (default), read/write a local Parquet cache and only
        fetch new bars from OANDA.
    """
    cached = (
        _read_cache(_cache_path(symbol, interval, asset_class)) if use_cache else None
    )

    if cached is not None and not cached.empty:
        # Drop last cached row — it may have been an incomplete candle.
        cached = cached.iloc[:-1]
        if len(cached) >= n_bars:
            fetch_start = cached.index[-1].to_pydatetime()
            tail = _fetch_tail(symbol, interval, asset_class, start_time=fetch_start)
            result = pd.concat([cached, tail]).sort_index()
            result = result[~result.index.duplicated(keep="last")]
        else:
            # Cache only covers a smaller window than requested, so tail-only
            # refresh cannot recover the missing older bars.
            result = _fetch_full_no_cache(symbol, interval, n_bars, asset_class)
    else:
        result = _fetch_full_no_cache(symbol, interval, n_bars, asset_class)

    # Trim to requested size
    result = result.iloc[-n_bars:]

    # Persist before applying closed_only filter
    if use_cache and not result.empty:
        _write_cache(_cache_path(symbol, interval, asset_class), result)

    if closed_only and not result.empty:
        now = datetime.now(tz=timezone.utc)
        last_open = result.index[-1]
        expected_close = last_open + pd.tseries.frequencies.to_offset(interval)
        if expected_close > now:
            result = result.iloc[:-1]

    result.attrs["symbol"] = symbol
    result.attrs["timeframe"] = interval
    return result


# ---------------------------------------------------------------------------
# Public: parallel multi-instrument fetch
# ---------------------------------------------------------------------------


def fetch_multi(
    tasks: list[tuple[str, str]],
    n_bars: int = 2000,
    asset_class: str = "forex",
    max_workers: int = 6,
    closed_only: bool = True,
    use_cache: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch candles for multiple (symbol, interval) pairs in parallel.

    Parameters
    ----------
    tasks:
        List of ``(symbol, interval)`` tuples, e.g.
        ``[("EUR_USD", "5min"), ("XAU_USD", "4h")]``.
    n_bars:
        Number of bars to fetch per task.
    asset_class:
        ``"forex"``, ``"indices"``, or ``"commodities"``.
    max_workers:
        Maximum concurrent requests (default 6).
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
            symbol,
            interval,
            n_bars=n_bars,
            asset_class=asset_class,
            closed_only=closed_only,
            use_cache=use_cache,
        )
        return task, df

    results: dict[tuple[str, str], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, df in pool.map(_fetch, tasks):
            results[key] = df
    return results


if __name__ == "__main__":
    df = fetch_klines("EUR_USD", "1D", limit=100, asset_class="forex")
    print(f"\n{len(df)} bars fetched, index: {df.index[0]} → {df.index[-1]}")
    fig = plot_market_structure(df)
    fig.show()
