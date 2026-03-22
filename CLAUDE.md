# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run main engine (loads btcusd.pkl, computes multi-timeframe trends and cluster signals)
uv run main.py

# Run tests
uv run pytest

# Lint / format imports
uv run ruff check --fix .
```

Use `uv` for package management (lockfile: `uv.lock`). Python 3.12 required.

## Architecture

All core logic lives in `ms_engine.py`. `main.py` is the entry point that wires it together. `binance_data.py` handles data fetching from Binance. `chart.py` renders candlestick charts with market-structure overlays via Plotly.

### Data flow

```
Binance API (binance_data.py)
    → detect_pivots()          # manual window scan, confirmed N bars later
    → compute_market_structure()  # state machine over pivot arrays → trend array
    → get_mtf_trend()          # forward-fill higher-TF trend to base index
    → compute_cluster_signals()   # 3-TF confluence → long/short boolean arrays
    → plot_market_structure()  # chart.py: candlestick chart with trend coloring + signals
```

### Key abstractions

- **`detect_pivots(high, low, pivot_length)`** — finds pivot highs/lows with `pivot_length`-bar confirmation delay using a manual window scan (no scipy dependency).
- **`MarketStructureState` / `update_market_structure()`** — bar-by-bar state machine tracking trend (1/−1/0), support, resistance, and last pivot. Trend flips when price breaks support/resistance levels.
- **`get_mtf_trend(df, rule, pivot_length, higher_tf_df=None)`** — computes trend on a higher timeframe and forward-fills to base index (no lookahead). Accepts pre-fetched higher-TF data via `higher_tf_df` to avoid resampling artefacts.
- **`ClusterState` / `compute_cluster_signals()`** — three-timeframe confluence detector. Long: high=up → med dips → recovers → dips → low dips → recovers = signal. Short is the mirror. Fires once per setup, resets on trend break.

### Timeframe convention

`main.py` uses three timeframes: `5min` (base), `30min` (medium), `4h` (high). The cluster signal logic assumes this high/med/low ordering.
