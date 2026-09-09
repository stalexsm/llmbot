"""Alembic environment rag-БД: применяет ревизии к файлу rag.db (ADR-0003)."""

import sqlite3

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from bot.rag.schema import metadata

config = context.config

target_metadata = metadata


def _load_sqlite_vec(connection: Connection) -> None:
    """Загрузить расширение sqlite-vec: ревизии создают vec0-таблицу."""
    import sqlite_vec

    raw = connection.connection.driver_connection
    if not isinstance(raw, sqlite3.Connection):
        raise RuntimeError("rag migrations require a sqlite3 driver connection")
    raw.enable_load_extension(True)
    sqlite_vec.load(raw)
    raw.enable_load_extension(False)


def include_object(
    obj: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object,
) -> bool:
    """Выкинуть vec0 и его теневые таблицы из автогенерации.

    Виртуальная таблица и служебные таблицы ``chunk_vectors*`` живут вне
    метаданных SQLAlchemy; без фильтра автогенерация считала бы их лишними
    и предлагала удалить.
    """
    return not (
        type_ == "table"
        and isinstance(name, str)
        and (name.startswith("chunk_vectors") or name.startswith("sqlite_"))
    )


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        message = "sqlalchemy.url is not configured for rag migrations"
        raise RuntimeError(message)
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        _load_sqlite_vec(connection)
        # render_as_batch: будущие ALTER в SQLite идут через пересоздание таблицы.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
