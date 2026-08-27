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
