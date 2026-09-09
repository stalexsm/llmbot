"""Схема БД чат-сессий в терминах SQLAlchemy Core (ADR-0003).

Метаданные — источник истины для ``alembic revision --autogenerate``:
новые ревизии создаются только автогенерацией против этой схемы.
Рантайм-хранилище работает через stdlib sqlite3 и схему не создаёт —
SQLAlchemy используется исключительно миграциями.
"""

from sqlalchemy import Column, Index, Integer, MetaData, String, Table, Text, text

metadata = MetaData()

messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("chat_id", Integer, nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"),
    ),
    # Окно истории: последние N сообщений одного чата (ORDER BY id DESC LIMIT N).
    Index("messages_chat_id_id", "chat_id", "id"),
)
