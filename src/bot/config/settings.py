"""Application settings loaded from environment variables and ``.env``.

Pydantic is used only at this external configuration boundary; everything
inside the application operates on plain typed dataclasses.
"""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    ollama_timeout_seconds: float = 120.0
    telegram_timeout_seconds: float = 30.0

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
