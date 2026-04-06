# AGENTS.md

## Setup

- Use `uv` for dependency management and commands.
- Python 3.12 is required.
- Install project and dev tools with `uv sync --dev`.

## Commands

- Run the app: `uv run main.py`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Lint and sort imports: `uv run ruff check --fix .`
- Format: `uv run ruff format .`

## Repo Layout

- `main.py`: entry point for multi-market signal generation.
- `ms_engine.py`: core market-structure and cluster logic.
- `binance_data.py`: Binance market data fetcher.
- `tradfi_data.py`: OANDA market data fetcher for forex, indices, and commodities.
- `chart.py`: Plotly candlestick chart rendering.

## Notes

- `tradfi_data.py` requires `OANDA_API` in `.env` or the environment.
- Set `OANDA_ENV=live` for live OANDA accounts; otherwise practice is used.
- The cluster workflow is data fetch -> pivots -> market structure -> MTF trend -> cluster signals -> charting.
