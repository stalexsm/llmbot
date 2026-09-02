"""Настройки бенчмарка: конфигурация агента без секретов Telegram.

Живой прогон бенчмарка не поднимает Telegram, поэтому токен бота здесь
необязателен; остальные настройки (Ollama, агент, метрики) наследуются.
"""

from pydantic import SecretStr

from bot.config.settings import Settings


class BenchmarkSettings(Settings):
    """Settings без обязательного Telegram-токена."""

    telegram_bot_token: SecretStr = SecretStr("")
