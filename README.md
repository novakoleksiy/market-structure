# market-structure

Multi-market market-structure signal generation across Binance and OANDA data.

## Setup

- Python 3.12
- `uv sync --dev`

## Commands

- Run the app: `uv run main.py`
- Run a scheduled source+cluster job and send Telegram alerts for newly persisted signals: `uv run main.py run-source-cluster --source oanda --cluster C1 --db-path .cache/signals.sqlite3`
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
- Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` or the environment to enable scheduled signal alerts.
