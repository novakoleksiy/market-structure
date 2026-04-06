# market-structure

Multi-market market-structure signal generation across Binance and OANDA data.

## Setup

- Python 3.12
- `uv sync --dev`

## Commands

- Run the app: `uv run main.py`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Lint and sort imports: `uv run ruff check --fix .`
- Format: `uv run ruff format .`

## Notes

- `main.py` is the entry point.
- `ms_engine.py` contains the core pivot, trend, and cluster logic.
- `binance_data.py` fetches Binance data.
- `tradfi_data.py` fetches OANDA data for forex, indices, and commodities.
- Set `OANDA_API` in `.env` or the environment for OANDA-backed fetches.
