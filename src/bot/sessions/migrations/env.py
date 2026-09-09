"""Alembic environment БД чат-сессий: применяет ревизии к файлу chats.db (ADR-0003)."""

from alembic import context
from sqlalchemy import create_engine, pool

from bot.sessions.schema import metadata

config = context.config

target_metadata = metadata


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        message = "sqlalchemy.url is not configured for chat session migrations"
        raise RuntimeError(message)
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        # render_as_batch: будущие ALTER в SQLite идут через пересоздание таблицы.
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
