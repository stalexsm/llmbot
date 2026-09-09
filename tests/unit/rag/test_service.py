"""Unit-тесты RAG-сервиса: pipeline индексации и поиска на фейковых эмбеддингах."""

from pathlib import Path

import pytest
import structlog.stdlib

from bot.application.errors import (
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.inference.embeddings import EmbeddingRequest, EmbeddingResponse
from bot.rag.migrations import apply_migrations
from bot.rag.models import DocumentKind
from bot.rag.service import RagService
from bot.rag.store import RagStore
from tests.fakes import MockEmbeddingProvider

OWNER_A = TelegramUserId(1)
OWNER_B = TelegramUserId(2)
REQUEST_ID = RequestId("rag-test")


def make_document(sections: int) -> str:
    """Документ из нескольких разделов: гарантированно больше одного чанка."""
    parts = ["Отпуск и отгулы", ""]
    for number in range(sections):
        if number == 0:
            parts.append(
                "Ежегодный отпуск составляет 28 календарных дней. Его можно разделить\n"
                "на части, если хотя бы одна часть не короче 14 дней. Сколько дней\n"
                "отпуска положено за стаж — см. таблицу ниже."
            )
        else:
            body = " ".join(f"слово{index}" for index in range(80))
            parts.append(f"Раздел {number}\n\n{body}")
        parts.append("")
    return "\n".join(parts).strip()


SMALL_DOCUMENT = make_document(sections=3)


def make_service(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    embeddings: MockEmbeddingProvider | None = None,
    **overrides: float | int,
) -> RagService:
    database = tmp_path / "rag.db"
    apply_migrations(database)
    params: dict[str, object] = {
        "chunk_target_chars": 900,
        "chunk_overlap_chars": 150,
        "top_k": 5,
        "overfetch": 4,
        "min_similarity": 0.35,
        "max_file_bytes": 20 * 1024 * 1024,
        "max_text_chars": 200_000,
        "max_chunks": 300,
    }
    params.update(overrides)
    return RagService(
        store=RagStore(database=database, logger=logger),
        embeddings=embeddings or MockEmbeddingProvider(),
        model=ModelId("test-embed"),
        logger=logger,
        **params,  # type: ignore[arg-type]
    )


async def test_index_document_then_search_finds_content(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Хэш-фейковка эмбеддингов детерминирована, но порогу реальной модели
    # не соответствует: проверяем механизм поиска без порога (порог —
    # отдельный тест ниже).
    service = make_service(tmp_path, logger, min_similarity=0.0)

    indexed = await service.index_document(
        REQUEST_ID, OWNER_A, "handbook.txt", SMALL_DOCUMENT.encode()
    )

    assert indexed.chunk_count >= 2
    assert indexed.document.name == "handbook.txt"
    assert indexed.document.kind == DocumentKind.TXT
    assert indexed.document.owner_id == OWNER_A
    hits = await service.search(REQUEST_ID, OWNER_A, "сколько дней отпуска")

    assert hits
    assert all(hit.document_name == "handbook.txt" for hit in hits)
    assert any("отпуск" in hit.text.lower() for hit in hits)


async def test_indexing_passes_chunk_texts_and_model_to_provider(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    embeddings = MockEmbeddingProvider()
    service = make_service(tmp_path, logger, embeddings=embeddings)

    await service.index_document(REQUEST_ID, OWNER_A, "handbook.txt", SMALL_DOCUMENT.encode())

    assert len(embeddings.requests) == 1
    request = embeddings.requests[0]
    assert request.model == ModelId("test-embed")
    assert request.request_id == REQUEST_ID
    assert len(request.texts) >= 2
    assert "отпуск" in " ".join(request.texts)


async def test_owner_corpora_are_isolated(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, min_similarity=0.0)
    await service.index_document(REQUEST_ID, OWNER_A, "a.txt", SMALL_DOCUMENT.encode())
    await service.index_document(
        REQUEST_ID, OWNER_B, "b.txt", "Совершенно другой текст про номера счетов.".encode()
    )

    hits_a = await service.search(REQUEST_ID, OWNER_A, "сколько дней отпуска")

    # Поиск владельца A никогда не возвращает чанки B (ADR-0002).
    assert hits_a
    assert all(hit.document_name == "a.txt" for hit in hits_a)


async def test_same_name_is_replaced(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    service = make_service(tmp_path, logger)
    await service.index_document(REQUEST_ID, OWNER_A, "doc.txt", SMALL_DOCUMENT.encode())

    indexed = await service.index_document(
        REQUEST_ID, OWNER_A, "doc.txt", "Одна новая версия документа.".encode()
    )

    assert indexed.chunk_count == 1
    hits = await service.search(REQUEST_ID, OWNER_A, "новая версия документа")
    assert [hit.text for hit in hits] == ["Одна новая версия документа."]


async def test_oversized_file_is_rejected_before_processing(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    embeddings = MockEmbeddingProvider()
    service = make_service(tmp_path, logger, embeddings=embeddings, max_file_bytes=10)

    with pytest.raises(DocumentTooLargeError):
        await service.index_document(REQUEST_ID, OWNER_A, "big.txt", b"0" * 11)

    # Отсечение по лимиту файла происходит до обращения к модели.
    assert embeddings.requests == []


async def test_oversized_text_is_rejected(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, max_text_chars=50)

    with pytest.raises(DocumentTooLargeError):
        await service.index_document(
            REQUEST_ID, OWNER_A, "long.txt", ("абзац текста " * 20).encode()
        )


async def test_too_many_chunks_are_rejected(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    text = "\n\n".join(f"абзац номер {number} с текстом" for number in range(10))
    service = make_service(
        tmp_path, logger, max_chunks=2, chunk_target_chars=30, chunk_overlap_chars=5
    )

    with pytest.raises(DocumentTooLargeError):
        await service.index_document(REQUEST_ID, OWNER_A, "many.txt", text.encode())


async def test_empty_and_unsupported_documents_are_rejected(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)

    with pytest.raises(EmptyDocumentError):
        await service.index_document(REQUEST_ID, OWNER_A, "empty.txt", b"   \n")
    with pytest.raises(UnsupportedDocumentError):
        await service.index_document(REQUEST_ID, OWNER_A, "scan.pdf", b"%PDF-1.4 fake")


class WrongCountEmbeddingProvider(MockEmbeddingProvider):
    """Фейковка, возвращающая ноль векторов на любой запрос."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response = await super().embed(request)
        return EmbeddingResponse(request_id=response.request_id, embeddings=())


async def test_embedding_count_mismatch_is_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, embeddings=WrongCountEmbeddingProvider())

    with pytest.raises(EmbeddingError):
        await service.index_document(REQUEST_ID, OWNER_A, "doc.txt", SMALL_DOCUMENT.encode())


async def test_embedding_dimension_mismatch_is_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, embeddings=MockEmbeddingProvider(dimension=8))

    with pytest.raises(EmbeddingError):
        await service.index_document(REQUEST_ID, OWNER_A, "doc.txt", SMALL_DOCUMENT.encode())


async def test_empty_query_returns_nothing_without_model_call(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    embeddings = MockEmbeddingProvider()
    service = make_service(tmp_path, logger, embeddings=embeddings)

    assert await service.search(REQUEST_ID, OWNER_A, "   ") == ()
    assert embeddings.requests == []


async def test_similarity_threshold_filters_irrelevant_query(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, min_similarity=0.99)
    await service.index_document(REQUEST_ID, OWNER_A, "doc.txt", SMALL_DOCUMENT.encode())

    # Ниже порога близости — явное «ничего не найдено».
    assert await service.search(REQUEST_ID, OWNER_A, "полностью посторонний запрос") == ()
