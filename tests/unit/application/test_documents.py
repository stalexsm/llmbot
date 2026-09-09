"""Тесты DocumentService: фоновая индексация с сериализацией по владельцу."""

import asyncio
from pathlib import Path

import pytest
import structlog.stdlib

from bot.application.documents import DocumentService, DocumentUpload
from bot.application.errors import DocumentTooLargeError, EmptyDocumentError
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.inference.embeddings import EmbeddingRequest, EmbeddingResponse
from bot.rag.migrations import apply_migrations
from bot.rag.service import RagService
from bot.rag.store import RagStore
from tests.fakes import MockEmbeddingProvider

OWNER_A = TelegramUserId(1)
OWNER_B = TelegramUserId(2)


class SlowMockEmbeddingProvider(MockEmbeddingProvider):
    """Фейковка с задержкой и учётом параллелизма вызовов."""

    def __init__(self, delay_seconds: float) -> None:
        super().__init__()
        self._delay = delay_seconds
        self.concurrent = 0
        self.max_concurrent = 0

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self._delay)
        finally:
            self.concurrent -= 1
        return await super().embed(request)


def make_documents(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    embeddings: MockEmbeddingProvider | None = None,
) -> DocumentService:
    database = tmp_path / "rag.db"
    apply_migrations(database)
    rag = RagService(
        store=RagStore(database=database, logger=logger),
        embeddings=embeddings or MockEmbeddingProvider(),
        model=ModelId("test-embed"),
        chunk_target_chars=900,
        chunk_overlap_chars=150,
        top_k=5,
        overfetch=4,
        min_similarity=0.35,
        max_file_bytes=1000,
        max_text_chars=5000,
        max_chunks=10,
        logger=logger,
    )
    return DocumentService(rag, logger)


def upload(owner_id: TelegramUserId, name: str, content: bytes) -> DocumentUpload:
    return DocumentUpload(
        request_id=RequestId(f"req-{name}-{int(owner_id)}"),
        owner_id=owner_id,
        name=name,
        content=content,
    )


async def test_index_returns_result_and_document_becomes_listed(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    documents = make_documents(tmp_path, logger)

    result = await documents.index(upload(OWNER_A, "doc.txt", "привет мир".encode()))

    assert result.name == "doc.txt"
    assert result.chunk_count == 1
    assert [document.name for document in documents.list_documents(OWNER_A)] == ["doc.txt"]


async def test_same_owner_uploads_are_serialized(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    embeddings = SlowMockEmbeddingProvider(delay_seconds=0.03)
    documents = make_documents(tmp_path, logger, embeddings)

    await asyncio.gather(
        documents.index(upload(OWNER_A, "first.txt", "первый текст".encode())),
        documents.index(upload(OWNER_A, "second.txt", "второй текст".encode())),
    )

    assert embeddings.max_concurrent == 1
    assert len(documents.list_documents(OWNER_A)) == 2


async def test_different_owners_index_in_parallel(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    embeddings = SlowMockEmbeddingProvider(delay_seconds=0.05)
    documents = make_documents(tmp_path, logger, embeddings)

    await asyncio.gather(
        documents.index(upload(OWNER_A, "a.txt", "текст владельца A".encode())),
        documents.index(upload(OWNER_B, "b.txt", "текст владельца B".encode())),
    )

    assert embeddings.max_concurrent == 2
    assert len(documents.list_documents(OWNER_A)) == 1
    assert len(documents.list_documents(OWNER_B)) == 1


async def test_index_errors_propagate_to_caller(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    documents = make_documents(tmp_path, logger)

    with pytest.raises(EmptyDocumentError):
        await documents.index(upload(OWNER_A, "empty.txt", b"   \n  "))
    with pytest.raises(DocumentTooLargeError):
        await documents.index(upload(OWNER_A, "big.txt", b"x" * 2000))

    # Ошибочные загрузки корпус не меняют.
    assert documents.list_documents(OWNER_A) == ()


async def test_delete_unknown_document_raises(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    from bot.application.errors import DocumentNotFoundError

    documents = make_documents(tmp_path, logger)

    with pytest.raises(DocumentNotFoundError):
        documents.delete_document(OWNER_A, "missing.txt")


async def test_clear_documents_wipes_owner_corpus_only(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    documents = make_documents(tmp_path, logger)
    await documents.index(upload(OWNER_A, "a.txt", "текст владельца A".encode()))
    await documents.index(upload(OWNER_B, "b.txt", "текст владельца B".encode()))

    deleted = await documents.clear_documents(OWNER_A)

    assert deleted == 1
    assert documents.list_documents(OWNER_A) == ()
    assert [document.name for document in documents.list_documents(OWNER_B)] == ["b.txt"]


async def test_clear_waits_for_running_index(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    """Очистка берёт блокировку владельца: ждёт идущую индексацию.

    Иначе /clear на фоне фоновой индексации отчитался бы «корпус очищен»,
    а доехавший следом документ всё равно появился бы в корпусе.
    """
    embeddings = SlowMockEmbeddingProvider(delay_seconds=0.1)
    documents = make_documents(tmp_path, logger, embeddings)
    await documents.index(upload(OWNER_A, "old.txt", "старый текст".encode()))

    index_task = asyncio.create_task(
        documents.index(upload(OWNER_A, "new.txt", "новый текст".encode()))
    )
    await asyncio.sleep(0.02)  # индексация успела стартовать и держит блокировку
    deleted = await documents.clear_documents(OWNER_A)

    assert deleted == 2  # вычищен и прежний корпус, и доехавший документ
    assert documents.list_documents(OWNER_A) == ()
    await index_task
