"""Хранилище rag-БД: documents → chunks → vec0 (sqlite-vec, ADR-0003).

Один файл ``rag.db`` (вне Git): слои ``sessions/`` и ``rag/`` владеют каждый
своей схемой и соединением, расширение sqlite-vec грузится только на
rag-соединение. stdlib ``sqlite3``, синхронно; соединение открывается на
операцию и закрывается после неё. Повторная индексация того же имени
заменяет документ атомарно — старые документ, чанки и векторы удаляются,
новые появляются одной транзакцией. Сырой файл после индексации не хранится.

Схемой владеет alembic (``bot.rag.migrations``): миграции применяет
композиционный корень на старте, хранилище работает только с готовой схемой.
Сбой БД маппится на прикладное ``RagStorageError``.
"""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec
import structlog

from bot.application.errors import RagStorageError
from bot.domain.ids import DocumentId, TelegramUserId
from bot.rag.models import EMBEDDING_DIMENSION, Chunk, DocumentInfo, SearchHit

# Запас на ожидание блокировки: поиск во время замены должен видеть
# старый корпус, а не падать по «database is locked».
_BUSY_TIMEOUT_MS = 5000


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    """rowid вставленной строки; ``None`` после INSERT — сбой драйвера."""
    value = cursor.lastrowid
    if value is None:
        message = "INSERT did not return a rowid"
        raise RuntimeError(message)
    return value


class RagStore:
    """Persisted document corpus in a SQLite database with a vector index."""

    def __init__(self, database: Path, logger: structlog.stdlib.BoundLogger) -> None:
        self._database = database
        self._logger = logger.bind(component="rag_store")

    def replace_document(
        self,
        owner_id: TelegramUserId,
        name: str,
        kind: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> DocumentInfo:
        """Записать документ с чанками и векторами одной транзакцией.

        Существующий документ того же владельца с тем же именем заменяется:
        его чанки и векторы удаляются вместе с ним. Возвращаются сведения
        о документе (id, момент создания).
        """
        if len(chunks) != len(embeddings):
            message = "chunk and embedding counts differ"
            raise ValueError(message)
        for vector in embeddings:
            self._validate_dimension(vector)
        owner_key = int(owner_id)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._delete_document(connection, owner_key, name)
                cursor = connection.execute(
                    "INSERT INTO documents (owner_id, name, kind) VALUES (?, ?, ?)",
                    (owner_key, name, kind),
                )
                document_id = DocumentId(_lastrowid(cursor))
                for chunk, vector in zip(chunks, embeddings, strict=True):
                    chunk_cursor = connection.execute(
                        "INSERT INTO chunks (document_id, position, text, page)"
                        " VALUES (?, ?, ?, ?)",
                        (document_id, chunk.position, chunk.text, chunk.page),
                    )
                    connection.execute(
                        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (_lastrowid(chunk_cursor), self._serialize(vector)),
                    )
                created_at = connection.execute(
                    "SELECT created_at FROM documents WHERE id = ?", (document_id,)
                ).fetchone()[0]
        except (sqlite3.Error, OSError) as exc:
            self._logger.warning(
                "rag_storage_failed",
                owner_id=owner_key,
                operation="replace_document",
                status="error",
            )
            raise RagStorageError("Failed to write the document to the RAG database") from exc
        self._logger.info(
            "rag_document_replaced",
            owner_id=owner_key,
            document_id=document_id,
            chunks=len(chunks),
        )
        return DocumentInfo(
            id=document_id,
            owner_id=owner_id,
            name=name,
            kind=kind,
            created_at=str(created_at),
        )

    def search(
        self,
        owner_id: TelegramUserId,
        embedding: Sequence[float],
        *,
        top_k: int,
        overfetch: int,
        min_similarity: float,
    ) -> tuple[SearchHit, ...]:
        """Найти top-K чанков корпуса владельца по косинусной близости.

        Внутренний переопрос шире в ``overfetch`` раз: KNN по всей таблице
        векторов, затем фильтр по владельцу и отсечка до top-K. Ниже
        ``min_similarity`` чанки отбрасываются — поиск честно отвечает
        «ничего не найдено» пустым кортежем.
        """
        self._validate_dimension(embedding)
        limit = top_k * overfetch
        query = self._serialize(embedding)
        owner_key = int(owner_id)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT c.position, c.text, c.page, d.name, v.distance
                    FROM (
                        SELECT chunk_id, distance FROM chunk_vectors
                        WHERE embedding MATCH ? AND k = ?
                    ) AS v
                    JOIN chunks AS c ON c.id = v.chunk_id
                    JOIN documents AS d ON d.id = c.document_id
                    WHERE d.owner_id = ?
                    ORDER BY v.distance
                    LIMIT ?
                    """,
                    (query, limit, owner_key, top_k),
                ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            self._logger.warning(
                "rag_storage_failed",
                owner_id=owner_key,
                operation="search",
                status="error",
            )
            raise RagStorageError("Failed to search the RAG database") from exc
        hits = tuple(
            SearchHit(
                document_name=str(name),
                position=int(position),
                page=None if page is None else int(page),
                text=str(text),
                similarity=1.0 - float(distance),
            )
            for position, text, page, name, distance in rows
            if 1.0 - float(distance) >= min_similarity
        )
        self._logger.info(
            "rag_search_completed",
            owner_id=owner_key,
            hits=len(hits),
            top_similarity=hits[0].similarity if hits else None,
        )
        return hits

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Соединение на одну операцию; на успехе — commit, на сбое — rollback."""
        self._database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database)
        try:
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            # sqlite-vec грузится только на rag-соединение (ADR-0003).
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _validate_dimension(vector: Sequence[float]) -> None:
        """Вектор обязан совпадать с размерностью схемы vec0 (bge-m3: 1024)."""
        if len(vector) != EMBEDDING_DIMENSION:
            message = f"embedding dimension must be {EMBEDDING_DIMENSION}"
            raise ValueError(message)

    @staticmethod
    def _serialize(vector: Sequence[float]) -> bytes:
        """Вектор → бинарный формат sqlite-vec."""
        return sqlite_vec.serialize_float32([float(value) for value in vector])

    @staticmethod
    def _delete_document(connection: sqlite3.Connection, owner_key: int, name: str) -> None:
        """Удалить прежнюю версию документа владельца (если была)."""
        connection.execute(
            """
            DELETE FROM chunk_vectors WHERE chunk_id IN (
                SELECT c.id FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.owner_id = ? AND d.name = ?
            )
            """,
            (owner_key, name),
        )
        connection.execute(
            """
            DELETE FROM chunks WHERE document_id IN (
                SELECT id FROM documents WHERE owner_id = ? AND name = ?
            )
            """,
            (owner_key, name),
        )
        connection.execute(
            "DELETE FROM documents WHERE owner_id = ? AND name = ?",
            (owner_key, name),
        )
