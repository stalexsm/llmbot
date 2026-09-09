"""Схема rag-БД в терминах SQLAlchemy Core (ADR-0003).

Метаданные — источник истины для ``alembic revision --autogenerate``:
ревизии создаются только автогенерацией против этой схемы (секция
``[rag]`` в ``alembic.ini``). Виртуальная таблица векторов vec0 в
метаданных невыразима — она добавляется ``op.execute`` внутри ревизии.

Рантайм-хранилище работает через stdlib sqlite3 и схему не создаёт —
SQLAlchemy используется исключительно миграциями.
"""

from sqlalchemy import Column, Index, Integer, MetaData, String, Table, Text, text

metadata = MetaData()

# documents → chunks → chunk_vectors (vec0, создаётся миграцией).
documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    # Владелец документа — TelegramUserId (ADR-0002); изоляция корпусов.
    Column("owner_id", Integer, nullable=False),
    Column("name", String, nullable=False),
    Column("kind", String, nullable=False),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"),
    ),
    # Повторная индексация того же имени заменяет документ атомарно.
    Index("documents_owner_id_name", "owner_id", "name", unique=True),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer, nullable=False),
    # Порядковый номер чанка внутри документа.
    Column("position", Integer, nullable=False),
    Column("text", Text, nullable=False),
    # Страница источника, когда формат её знает (PDF); для .txt — NULL.
    Column("page", Integer),
    Index("chunks_document_id_position", "document_id", "position"),
)
