"""Тесты жизненного цикла документа: стадии прогресса, список, удаление."""

from pathlib import Path

import pytest
import structlog.stdlib

from bot.application.errors import DocumentNotFoundError
from bot.application.progress import IndexingStage
from bot.domain.ids import RequestId, TelegramUserId
from tests.fakes import MockEmbeddingProvider
from tests.unit.rag.test_service import SMALL_DOCUMENT, make_service

OWNER_A = TelegramUserId(1)
OWNER_B = TelegramUserId(2)
REQUEST_ID = RequestId("lifecycle-test")


class RecordingProgress:
    """Наблюдатель стадий, запоминающий порядок и аргументы."""

    def __init__(self) -> None:
        self.stages: list[tuple[IndexingStage, int]] = []

    async def on_stage(self, stage: IndexingStage, *, chunks: int = 0) -> None:
        self.stages.append((stage, chunks))


async def test_index_reports_stages_in_pipeline_order(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)
    progress = RecordingProgress()

    await service.index_document(
        REQUEST_ID, OWNER_A, "handbook.txt", SMALL_DOCUMENT.encode(), progress=progress
    )

    assert [stage for stage, _ in progress.stages] == [
        IndexingStage.EXTRACTING,
        IndexingStage.CHUNKED,
        IndexingStage.EMBEDDING,
    ]
    chunked = dict(progress.stages)[IndexingStage.CHUNKED]
    assert chunked >= 1


async def test_index_without_progress_observer_succeeds(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)

    indexed = await service.index_document(
        REQUEST_ID, OWNER_A, "handbook.txt", SMALL_DOCUMENT.encode()
    )

    assert indexed.chunk_count >= 1


async def test_list_documents_returns_owner_corpus_sorted(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)

    await service.index_document(REQUEST_ID, OWNER_A, "b.txt", "первый документ".encode())
    await service.index_document(REQUEST_ID, OWNER_A, "a.txt", "второй документ".encode())
    await service.index_document(REQUEST_ID, OWNER_B, "c.txt", "чужой документ".encode())

    names = [document.name for document in service.list_documents(OWNER_A)]

    assert names == ["a.txt", "b.txt"]


async def test_delete_document_removes_it_from_corpus_and_search(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger, min_similarity=0.0)
    await service.index_document(REQUEST_ID, OWNER_A, "handbook.txt", SMALL_DOCUMENT.encode())

    service.delete_document(OWNER_A, "handbook.txt")

    assert service.list_documents(OWNER_A) == ()
    # Чанки и эмбеддинги вычищены: поиск больше ничего не находит.
    hits = await service.search(REQUEST_ID, OWNER_A, "отпуск")
    assert hits == ()


async def test_delete_unknown_name_raises_not_found(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)

    with pytest.raises(DocumentNotFoundError):
        service.delete_document(OWNER_A, "missing.txt")


async def test_delete_is_scoped_to_owner(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_service(tmp_path, logger)
    await service.index_document(REQUEST_ID, OWNER_A, "shared.txt", "общий текст".encode())

    with pytest.raises(DocumentNotFoundError):
        service.delete_document(OWNER_B, "shared.txt")

    assert [document.name for document in service.list_documents(OWNER_A)] == ["shared.txt"]


async def test_default_embedding_mock_is_used(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    """Помощник make_service совместим: индексация на фейковых эмбеддингах."""
    service = make_service(tmp_path, logger, embeddings=MockEmbeddingProvider())

    indexed = await service.index_document(
        REQUEST_ID, OWNER_A, "doc.txt", "текст документа".encode()
    )

    assert indexed.chunk_count == 1
