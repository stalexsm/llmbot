"""Unit-тесты хранилища rag-БД: реальный SQLite + sqlite-vec, tmp-файл."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
import sqlite_vec
import structlog.stdlib

from bot.application.errors import RagStorageError
from bot.domain.ids import TelegramUserId
from bot.rag.migrations import apply_migrations
from bot.rag.models import EMBEDDING_DIMENSION, Chunk, DocumentKind, SearchHit
from bot.rag.store import RagStore

OWNER_A = TelegramUserId(1)
OWNER_B = TelegramUserId(2)


def unit_vector(index: int) -> list[float]:
    """Ортогональные направления: близость между разными осями — нулевая."""
    values = [0.0] * EMBEDDING_DIMENSION
    values[index] = 1.0
    return values


def make_store(
    database: Path,
    logger: structlog.stdlib.BoundLogger,
) -> RagStore:
    # Как в композиционном корне: схема — миграции alembic до первой операции.
    apply_migrations(database)
    return RagStore(database=database, logger=logger)


def fetch_counts(database: Path) -> dict[str, int]:
    """Число строк документов, чанков и векторов: проверка атомарной замены."""
    connection = sqlite3.connect(database)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("documents", "chunks", "chunk_vectors")
        }
    finally:
        connection.close()


def test_replace_document_returns_document_info(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)

    info = store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="текст чанка")],
        [unit_vector(0)],
    )

    assert info.id >= 1
    assert info.owner_id == OWNER_A
    assert info.name == "rules.txt"
    assert info.kind == DocumentKind.TXT
    datetime.fromisoformat(info.created_at.replace("Z", "+00:00"))  # parseable ISO timestamp


def test_search_finds_indexed_chunk_with_source(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)
    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="сколько дней отпуска"), Chunk(position=1, text="погода")],
        [unit_vector(0), unit_vector(1)],
    )

    hits = store.search(
        OWNER_A,
        unit_vector(0),
        top_k=5,
        overfetch=4,
        min_similarity=0.0,
    )

    assert len(hits) == 2
    best = hits[0]
    assert isinstance(best, SearchHit)
    assert best.document_name == "rules.txt"
    assert best.position == 0
    assert best.page is None
    assert best.text == "сколько дней отпуска"
    assert best.similarity == pytest.approx(1.0, abs=1e-6)
    # Второй чанк ортогонален запросу: близость около нуля, он в хвосте.
    assert hits[1].similarity == pytest.approx(0.0, abs=1e-6)


def test_same_name_replaces_document_atomically(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    database = tmp_path / "rag.db"
    store = make_store(database, logger)
    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=index, text=f"старая версия {index}") for index in range(3)],
        [unit_vector(index) for index in range(3)],
    )

    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=index, text=f"новая версия {index}") for index in range(2)],
        [unit_vector(index) for index in range(2)],
    )

    # Одна строка документа, чанки и векторы только новой версии.
    assert fetch_counts(database) == {"documents": 1, "chunks": 2, "chunk_vectors": 2}
    hits = store.search(
        OWNER_A,
        unit_vector(0),
        top_k=10,
        overfetch=4,
        min_similarity=0.0,
    )
    assert all("новая версия" in hit.text for hit in hits)
    assert not any("старая версия" in hit.text for hit in hits)


def test_owner_corpora_are_isolated(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path / "rag.db", logger)
    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="документ владельца A")],
        [unit_vector(0)],
    )
    store.replace_document(
        OWNER_B,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="документ владельца B")],
        [unit_vector(1)],
    )

    hits_a = store.search(
        OWNER_A,
        unit_vector(0),
        top_k=5,
        overfetch=4,
        min_similarity=0.0,
    )
    hits_b = store.search(
        OWNER_B,
        unit_vector(1),
        top_k=5,
        overfetch=4,
        min_similarity=0.0,
    )

    # Поиск владельца A никогда не возвращает чанки B и наоборот (ADR-0002).
    assert [hit.text for hit in hits_a] == ["документ владельца A"]
    assert [hit.text for hit in hits_b] == ["документ владельца B"]


def test_search_returns_top_k_ordered_by_similarity(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)
    diagonal = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    diagonal[1] = 1.0  # (1, 1, 0, …): близость к оси 0 ≈ 1/√2
    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="точное совпадение"), Chunk(position=1, text="частичное")],
        [unit_vector(0), diagonal],
    )

    hits = store.search(
        OWNER_A,
        unit_vector(0),
        top_k=1,
        overfetch=4,
        min_similarity=0.0,
    )

    assert len(hits) == 1
    assert hits[0].text == "точное совпадение"


def test_below_threshold_returns_nothing(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)
    store.replace_document(
        OWNER_A,
        "rules.txt",
        DocumentKind.TXT,
        [Chunk(position=0, text="отпуск")],
        [unit_vector(0)],
    )

    # Ортогональный запрос не проходит порог — «ничего не найдено».
    assert (
        store.search(
            OWNER_A,
            unit_vector(1),
            top_k=5,
            overfetch=4,
            min_similarity=0.35,
        )
        == ()
    )


def test_search_empty_corpus_returns_nothing(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)

    assert (
        store.search(
            OWNER_A,
            unit_vector(0),
            top_k=5,
            overfetch=4,
            min_similarity=0.0,
        )
        == ()
    )


def test_clear_documents_removes_only_owner_corpus(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)
    for owner in (OWNER_A, OWNER_B):
        store.replace_document(
            owner,
            "rules.txt",
            DocumentKind.TXT,
            [Chunk(position=0, text=f"документ владельца {int(owner)}")],
            [unit_vector(0)],
        )
    store.replace_document(
        OWNER_A,
        "guide.md",
        DocumentKind.MD,
        [Chunk(position=0, text="второй документ владельца A")],
        [unit_vector(1)],
    )

    deleted = store.clear_documents(OWNER_A)

    assert deleted == 2
    assert fetch_counts(tmp_path / "rag.db") == {"documents": 1, "chunks": 1, "chunk_vectors": 1}
    assert store.list_documents(OWNER_A) == ()
    assert [document.name for document in store.list_documents(OWNER_B)] == ["rules.txt"]


def test_clear_empty_corpus_returns_zero(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)

    assert store.clear_documents(OWNER_A) == 0


def test_corrupt_database_maps_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    database = tmp_path / "rag.db"
    database.write_bytes(b"this is definitely not a sqlite database")
    # Без apply_migrations: миграция по битому файлу невозможна, хранилище
    # обязано маппить сбой на прикладное исключение на любой операции.
    store = RagStore(database=database, logger=logger)

    with pytest.raises(RagStorageError):
        store.replace_document(
            OWNER_A,
            "rules.txt",
            DocumentKind.TXT,
            [Chunk(position=0, text="текст")],
            [unit_vector(0)],
        )
    with pytest.raises(RagStorageError):
        store.search(OWNER_A, unit_vector(0), top_k=5, overfetch=4, min_similarity=0.0)


def test_unopenable_database_maps_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("не каталог", encoding="utf-8")
    store = RagStore(database=blocker / "rag.db", logger=logger)

    with pytest.raises(RagStorageError):
        store.replace_document(
            OWNER_A,
            "rules.txt",
            DocumentKind.TXT,
            [Chunk(position=0, text="текст")],
            [unit_vector(0)],
        )
    with pytest.raises(RagStorageError):
        store.search(OWNER_A, unit_vector(0), top_k=5, overfetch=4, min_similarity=0.0)


def test_wrong_dimension_vector_is_rejected(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "rag.db", logger)

    with pytest.raises(ValueError):
        store.replace_document(
            OWNER_A,
            "rules.txt",
            DocumentKind.TXT,
            [Chunk(position=0, text="текст")],
            [[1.0, 2.0, 3.0]],
        )
    with pytest.raises(ValueError):
        store.search(OWNER_A, [1.0, 2.0, 3.0], top_k=5, overfetch=4, min_similarity=0.0)
