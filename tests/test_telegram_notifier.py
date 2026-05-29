import urllib.error
from datetime import UTC, datetime

import pytest

from signal_engine import Signal
from telegram_notifier import (
    TelegramConfig,
    format_signal_summary,
    load_telegram_config,
    send_message,
)


@pytest.fixture(autouse=True)
def isolate_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr("telegram_notifier._ENV_FILE", tmp_path / ".env")


def test_load_telegram_config_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert load_telegram_config() is None


def test_load_telegram_config_requires_both_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        load_telegram_config()


def test_format_signal_summary_batches_signals():
    signals = [
        Signal(
            "EUR_USD",
            "C3",
            "short",
            datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
            1.2345,
            "oanda",
        ),
        Signal(
            "GBP_USD",
            "C3",
            "long",
            datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
            1.2712,
            "oanda",
        ),
    ]

    message = format_signal_summary(signals, "oanda", "C3")

    assert "New cluster signals" in message
    assert "Source: oanda" in message
    assert "Cluster: C3" in message
    assert "Bar: 2024-01-02 00:00 UTC" in message
    assert "SHORT EUR_USD" in message
    assert "LONG  GBP_USD" in message


def test_send_message_retries_until_success(monkeypatch):
    attempts: list[int] = []
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    monkeypatch.setattr("telegram_notifier.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("telegram_notifier.time.sleep", sleeps.append)

    send_message(
        "hello",
        TelegramConfig(
            bot_token="token",
            chat_id="chat",
            timeout_seconds=5.0,
            max_retries=3,
            retry_delay_seconds=1.5,
        ),
    )

    assert attempts == [5.0, 5.0, 5.0]
    assert sleeps == [1.5, 1.5]


def test_send_message_raises_after_retry_exhaustion(monkeypatch):
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("still failing")

    monkeypatch.setattr("telegram_notifier.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("telegram_notifier.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="attempt 2/2"):
        send_message(
            "hello",
            TelegramConfig(
                bot_token="token",
                chat_id="chat",
                timeout_seconds=5.0,
                max_retries=2,
                retry_delay_seconds=0.25,
            ),
        )

    assert sleeps == [0.25]
