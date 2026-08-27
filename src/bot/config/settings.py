"""Application settings loaded from environment variables and ``.env``.

Pydantic is used only at this external configuration boundary; everything
inside the application operates on plain typed dataclasses.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from bot.domain.ids import TelegramChatId


class Settings(BaseSettings):
    """Typed application configuration.

    Instantiation fails fast when required variables are missing or invalid.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: SecretStr
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    # Allowlist чатов через запятую: пусто — бот отвечает всем,
    # заполнен — сообщения из чужих чатов молча игнорируются.
    telegram_allowed_chat_ids: Annotated[frozenset[TelegramChatId], NoDecode] = frozenset()

    @field_validator("telegram_allowed_chat_ids", mode="before")
    @classmethod
    def _parse_allowed_chat_ids(cls, value: object) -> object:
        """Разобрать env-строку «123, 456».

        ``NoDecode`` отключает JSON-парсинг сложных типов в pydantic-settings;
        конвертацию элементов в int выполняет pydantic — невалидный id падает
        с ошибкой валидации на старте.
        """
        if not isinstance(value, str):
            return value
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]

    ollama_timeout_seconds: float = 120.0
    telegram_timeout_seconds: float = 30.0

    # Агентный цикл: лимит шагов и ограничения инструмента exec.
    agent_max_steps: int = Field(default=10, ge=1)
    agent_exec_timeout_seconds: float = Field(default=60.0, gt=0)
    agent_exec_max_output_chars: int = Field(default=4000, ge=200)

    # Окно истории чат-сессии: сколько последних сообщений уходит в запрос.
    agent_history_max_messages: int = Field(default=20, ge=1)

    # Каталог скиллов: markdown-файлы с инструкциями; индекс собирается на старте.
    agent_skills_directory: Path = Path("skills")

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
