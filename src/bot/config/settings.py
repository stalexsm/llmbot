"""Application settings loaded from environment variables and ``.env``.

Pydantic is used only at this external configuration boundary; everything
inside the application operates on plain typed dataclasses.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # Режим размышлений модели (think). Маленьким reasoning-моделям вроде
    # qwen3 он нужен: без него они могут молча выдавать пустой ответ
    # на многошаговых инструментальных задачах.
    # Моделям без поддержки thinking оставь false.
    ollama_think: bool = False

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
    # Таймаут запросов к /api/embed (OLLAMA_EMBED_TIMEOUT_SECONDS).
    ollama_embed_timeout_seconds: float = 120.0

    # Агентный цикл: лимит шагов и ограничения инструмента exec.
    agent_max_steps: int = Field(default=10, ge=1)
    agent_exec_timeout_seconds: float = Field(default=60.0, gt=0)
    agent_exec_max_output_chars: int = Field(default=4000, ge=200)

    # Окно истории чат-сессии: сколько последних сообщений уходит в запрос.
    agent_history_max_messages: int = Field(default=20, ge=1)

    # Компакция старших tool-выводов в агентном цикле: выводы шагов старше
    # последних K схлопываются в сигнатуру «команда → ok/ошибка». Записанная
    # чат-сессия не меняется. 0 — выключено.
    agent_compaction_keep_steps: int = Field(default=3, ge=0)

    # Каталог скиллов: markdown-файлы с инструкциями; индекс собирается на старте.
    agent_skills_directory: Path = Path("skills")

    # Учёт токенов: цены input/output за 1M токенов (в валюте учёта) для
    # оценки стоимости в метриках; по умолчанию 0 — стоимость считается нулевой.
    metrics_input_price_per_mtok: float = Field(default=0.0, ge=0)
    metrics_output_price_per_mtok: float = Field(default=0.0, ge=0)

    # --- RAG: индексация документов и поиск (модель bge-m3, 1024 измерения) ---

    # Модель эмбеддингов Ollama; скачивается отдельно: ollama pull bge-m3.
    ollama_embed_model: str = "bge-m3"

    # Чанкинг: целевой размер чанка в символах и overlap между соседними
    # чанками. Overlap обязан быть меньше целевого размера.
    rag_chunk_target_chars: int = Field(default=900, ge=100)
    rag_chunk_overlap_chars: int = Field(default=150, ge=0)

    # Поиск: сколько чанков возвращается (top-K), во сколько раз шире
    # внутренний переопрос перед фильтром по владельцу и минимальная
    # косинусная близость — ниже порога поиск отвечает «ничего не найдено».
    # Порог откалиброван на evaluation-датасете (``python -m bot.evaluation``):
    # худшее попадание 0.578, лучшее ложное срабатывание 0.437 — середина
    # разделяющей полосы 0.51, зафиксировано с запасом вниз.
    rag_search_top_k: int = Field(default=5, ge=1)
    rag_search_overfetch: int = Field(default=4, ge=1)
    rag_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)

    # Лимиты индексации: размер файла, извлечённого текста и число чанков.
    rag_max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    rag_max_text_chars: int = Field(default=200_000, ge=1)
    rag_max_chunks: int = Field(default=300, ge=1)

    @model_validator(mode="after")
    def _validate_rag_chunking(self) -> "Settings":
        """Overlap без отступа от целевого размера чанка лишает чанки нового содержимого."""
        if self.rag_chunk_overlap_chars >= self.rag_chunk_target_chars:
            raise ValueError("rag_chunk_overlap_chars must be less than rag_chunk_target_chars")
        return self

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
