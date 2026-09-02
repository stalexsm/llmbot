"""Настройки: allowlist чатов читается из env через запятую."""

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from bot.config.settings import Settings
from bot.domain.ids import TelegramChatId


def make_settings(monkeypatch: MonkeyPatch, raw_value: str | None) -> Settings:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    if raw_value is None:
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", raw_value)
    # ``_env_file=None`` отключает чтение локального ``.env``: тесты
    # проверяют только переменные окружения и не зависят от содержимого
    # рабочего каталога разработчика.
    return Settings(_env_file=None)


def test_allowlist_missing_means_empty(monkeypatch: MonkeyPatch) -> None:
    assert make_settings(monkeypatch, None).telegram_allowed_chat_ids == frozenset()


def test_allowlist_empty_string_means_empty(monkeypatch: MonkeyPatch) -> None:
    assert make_settings(monkeypatch, "").telegram_allowed_chat_ids == frozenset()


def test_allowlist_comma_separated_ids(monkeypatch: MonkeyPatch) -> None:
    settings = make_settings(monkeypatch, "123, 456")

    assert settings.telegram_allowed_chat_ids == frozenset(
        {TelegramChatId(123), TelegramChatId(456)}
    )


def test_allowlist_invalid_id_fails_fast(monkeypatch: MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        make_settings(monkeypatch, "123, oops")


def test_think_missing_means_disabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    assert Settings(_env_file=None).ollama_think is False


@pytest.mark.parametrize("raw", ["true", "1"])
def test_think_enabled_from_env(monkeypatch: MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OLLAMA_THINK", raw)
    assert Settings(_env_file=None).ollama_think is True


def test_metrics_prices_default_to_zero(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("METRICS_INPUT_PRICE_PER_MTOK", raising=False)
    monkeypatch.delenv("METRICS_OUTPUT_PRICE_PER_MTOK", raising=False)

    settings = Settings(_env_file=None)

    assert settings.metrics_input_price_per_mtok == 0.0
    assert settings.metrics_output_price_per_mtok == 0.0


def test_metrics_prices_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("METRICS_INPUT_PRICE_PER_MTOK", "0.5")
    monkeypatch.setenv("METRICS_OUTPUT_PRICE_PER_MTOK", "1.5")

    settings = Settings(_env_file=None)

    assert settings.metrics_input_price_per_mtok == 0.5
    assert settings.metrics_output_price_per_mtok == 1.5


def test_metrics_negative_price_fails_fast(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("METRICS_INPUT_PRICE_PER_MTOK", "-1")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
