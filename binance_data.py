"""Fetch OHLCV kline data from Binance public REST API."""

import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

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
    market: str = "spot",
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
        Number of bars to fetch (max 1000 per request).
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

    return df.drop(columns="close_time")


def fetch_klines_full(
    symbol: str = "BTCUSDT",
    interval: str = "5min",
    n_bars: int = 5000,
    market: str = "spot",
    closed_only: bool = True,
) -> pd.DataFrame:
    """Fetch more than 1000 bars by paginating Binance kline requests.

    Parameters
    ----------
    symbol:
        Binance trading pair.
    interval:
        Timeframe in pandas notation.
    n_bars:
        Total number of bars to retrieve (rounded up to nearest 1000).
    market:
        ``"spot"``, ``"futures-usdt"``, or ``"futures-coin"``.
    closed_only:
        If ``True`` (default), drop the current still-open bar from the result.
    """
    chunks: list[pd.DataFrame] = []
    end_time: datetime | None = None
    remaining = n_bars

    while remaining > 0:
        limit = min(remaining, 1000)
        chunk = fetch_klines(
            symbol,
            interval,
            limit=limit,
            end_time=end_time,
            market=market,
            closed_only=closed_only,
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        if len(chunk) < limit:
            break
        # Move end_time back one bar before the oldest in this chunk
        oldest_ms = int(chunk.index[0].timestamp() * 1000) - 1
        end_time = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return pd.concat(reversed(chunks)).sort_index().drop_duplicates()


if __name__ == "__main__":
    # USDT-margined perpetual (most liquid BTC perp on Binance)
    df = fetch_klines("BTCUSDT", "1W", limit=100, market="futures-usdt")
    print(df.tail())
    print(f"\n{len(df)} bars fetched, index: {df.index[0]} → {df.index[-1]}")
