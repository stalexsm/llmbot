"""Миграции схемы БД чат-сессий (ADR-0003).

Схема меняется только миграциями alembic; новая ревизия создаётся
исключительно автогенерацией:
``uv run alembic revision --autogenerate -m "..." --rev-id <имя>``
против метаданных ``bot.sessions.schema``. Композиционный корень применяет
неприменённые миграции один раз на старте, до первой операции хранилища;
само хранилище схему не создаёт и не проверяет.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIRECTORY = Path(__file__).parent

CHAT_DATABASE_PATH = Path(".data/chats.db")


def apply_migrations(database: Path) -> None:
    """Применить неприменённые миграции к файлу БД (идемпотентно).

    Сбой миграции не маппится в прикладные исключения: это сбой старта,
    процесс падает до начала работы (fail fast).
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    command.upgrade(config, "head")
