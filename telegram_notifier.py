"""Telegram notifications for newly persisted scheduled signals."""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from signal_engine import Signal

_ENV_FILE = Path(__file__).resolve().parent / ".env"
_SEND_MESSAGE_URL = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


def _get_env_setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    if not _ENV_FILE.exists():
        return None

    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None


def load_telegram_config() -> TelegramConfig | None:
    """Return Telegram config from env or .env when configured."""
    bot_token = _get_env_setting("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env_setting("TELEGRAM_CHAT_ID")
    timeout = _get_env_setting("TELEGRAM_TIMEOUT_SECONDS")
    max_retries = _get_env_setting("TELEGRAM_MAX_RETRIES")
    retry_delay = _get_env_setting("TELEGRAM_RETRY_DELAY_SECONDS")

    if not bot_token and not chat_id:
        return None
    if not bot_token or not chat_id:
        raise RuntimeError(
            "Telegram notifications require both TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID"
        )

    timeout_seconds = float(timeout) if timeout else 10.0
    return TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        timeout_seconds=timeout_seconds,
        max_retries=int(max_retries) if max_retries else 3,
        retry_delay_seconds=float(retry_delay) if retry_delay else 2.0,
    )


def format_signal_summary(signals: list[Signal], source: str, cluster: str) -> str:
    """Format one Telegram message for a scheduled source+cluster run."""
    if not signals:
        raise ValueError("Cannot format Telegram message without signals")

    latest_bar = max(signal.timestamp for signal in signals)
    lines = [
        "New cluster signals",
        f"Source: {source}",
        f"Cluster: {cluster}",
        f"Bar: {latest_bar.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    ordered = sorted(
        signals,
        key=lambda sig: (sig.timestamp, sig.symbol, sig.direction, sig.price),
    )
    for signal in ordered:
        lines.append(
            f"{signal.direction.upper():5s} {signal.symbol:12s} @ {signal.price:.5f}"
        )
    return "\n".join(lines)


def send_message(text: str, config: TelegramConfig) -> None:
    """Send one plain-text Telegram message with bounded retries."""
    payload = json.dumps({"chat_id": config.chat_id, "text": text}).encode("utf-8")
    attempts = max(1, config.max_retries)
    last_error: RuntimeError | None = None

    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            _SEND_MESSAGE_URL.format(token=config.bot_token),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            last_error = RuntimeError(
                f"Failed to send Telegram notification after attempt {attempt}/{attempts}: {exc}"
            )
        else:
            if body.get("ok"):
                return
            last_error = RuntimeError(
                "Telegram API rejected notification after attempt "
                f"{attempt}/{attempts}: {body.get('description', 'unknown error')}"
            )

        if attempt < attempts and config.retry_delay_seconds > 0:
            time.sleep(config.retry_delay_seconds)

    assert last_error is not None
    raise last_error


def notify_new_signals(signals: list[Signal], source: str, cluster: str) -> None:
    """Send a Telegram notification when a run persisted new signals."""
    if not signals:
        return

    config = load_telegram_config()
    if config is None:
        return

    send_message(format_signal_summary(signals, source, cluster), config)
