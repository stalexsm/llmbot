"""Unit-тесты инструмента search_documents: поиск по корпусу владельца.

Изоляция владельцев — обязательный тест (ADR-0002): поиск владельца A
никогда не возвращает чанки владельца B. Эмбеддинги — детерминированная
фейковка «ключевое слово — ось» из ``tests.fakes``: близость запроса
и чанков предсказуема без живой модели.
"""

import json
from pathlib import Path

import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.agent.progress import NullProgress
from bot.application.errors import EmbeddingError
from bot.application.search import SearchDocumentsTool
from bot.domain.ids import RequestId, TelegramUserId, ToolId
from bot.domain.tools import ExecutionContext, ToolCall
from bot.inference.embeddings import EmbeddingRequest, EmbeddingResponse
from bot.rag.service import RagService
from tests.fakes import make_rag_service

REQUEST_ID = RequestId("search-tool-test")
OWNER_A = TelegramUserId(1)
OWNER_B = TelegramUserId(2)
CONTEXT_A = ExecutionContext(owner_id=OWNER_A)
CONTEXT_B = ExecutionContext(owner_id=OWNER_B)
NO_OWNER = ExecutionContext()
PROGRESS = NullProgress()

DOCUMENT_A = (
    "Регламент отпусков\n"
    "\n"
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. Отпуск можно\n"
    "разделить на части, если хотя бы одна из них не короче 14 дней.\n"
)
DOCUMENT_B = (
    "Порядок командировок\n"
    "\n"
    "Суточные при командировке по России выплачиваются в размере 700 рублей\n"
    "за каждый день, включая дни в пути.\n"
)


class ExplodingEmbeddings:
    """Фейковка EmbeddingProvider: всегда падает ошибкой эмбеддинг-модели."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise self.error


def make_tool(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> SearchDocumentsTool:
    return SearchDocumentsTool(make_rag_service(tmp_path, logger), logger)


def make_tool_with(rag: RagService) -> SearchDocumentsTool:
    return SearchDocumentsTool(rag, structlog.stdlib.get_logger())


def search_call(query: str) -> ToolCall:
    return ToolCall(
        name=ToolId("search_documents"),
        arguments=json.dumps({"query": query}, ensure_ascii=False),
    )


def test_spec_describes_search_tool(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    spec = make_tool(tmp_path, logger).spec

    assert spec.name == ToolId("search_documents")
    assert spec.required == ("query",)
    assert [param.name for param in spec.parameters] == ["query"]
    assert all(param.type == "string" for param in spec.parameters)


async def test_hits_are_returned_with_sources(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    rag = make_rag_service(tmp_path, logger)
    await rag.index_document(REQUEST_ID, OWNER_A, "reglament.txt", DOCUMENT_A.encode())

    result = await make_tool_with(rag).execute(
        REQUEST_ID, search_call("сколько дней отпуска"), PROGRESS, CONTEXT_A
    )

    assert result.succeeded is True
    assert "Найдено фрагментов: 1" in result.content
    assert "Источник: reglament.txt" in result.content
    assert "28 календарных дней" in result.content


async def test_below_threshold_returns_explicit_not_found(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    rag = make_rag_service(tmp_path, logger)
    await rag.index_document(REQUEST_ID, OWNER_A, "reglament.txt", DOCUMENT_A.encode())
    tool = make_tool_with(rag)

    # Запрос мимо корпуса: ни одного известного ключевого слова.
    result = await tool.execute(
        REQUEST_ID, search_call("расписание поездов до Владивостока"), PROGRESS, CONTEXT_A
    )

    # «Не найдено» — данные для следующего шага модели, а не сбой.
    assert result.succeeded is True
    assert "ничего не найдено" in result.content.lower()


async def test_owner_isolation_a_never_sees_b_chunks(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    rag = make_rag_service(tmp_path, logger)
    await rag.index_document(REQUEST_ID, OWNER_A, "a-reglament.txt", DOCUMENT_A.encode())
    await rag.index_document(REQUEST_ID, OWNER_B, "b-trip.txt", DOCUMENT_B.encode())
    tool = make_tool_with(rag)

    result_a = await tool.execute(
        REQUEST_ID, search_call("сколько дней отпуска"), PROGRESS, CONTEXT_A
    )
    result_b = await tool.execute(
        REQUEST_ID, search_call("сколько платят в командировке"), PROGRESS, CONTEXT_B
    )

    # Каждый владелец видит только свой документ — по имени и содержимому.
    assert "a-reglament.txt" in result_a.content
    assert "b-trip.txt" not in result_a.content
    assert "командиров" not in result_a.content
    assert "b-trip.txt" in result_b.content
    assert "a-reglament.txt" not in result_b.content
    assert "28 календарных дней" not in result_b.content


async def test_empty_corpus_returns_not_found(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID, search_call("сколько дней отпуска"), PROGRESS, CONTEXT_A
    )

    assert result.succeeded is True
    assert "ничего не найдено" in result.content.lower()


async def test_invalid_arguments_return_error_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID,
        ToolCall(name=ToolId("search_documents"), arguments="не json"),
        PROGRESS,
        CONTEXT_A,
    )

    assert result.succeeded is False
    assert "query" in result.content


async def test_blank_query_is_invalid_arguments(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, search_call("   "), PROGRESS, CONTEXT_A)

    assert result.succeeded is False
    assert "query" in result.content


async def test_missing_owner_context_reports_unavailable(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, search_call("сколько дней отпуска"), PROGRESS, NO_OWNER)

    assert result.succeeded is False
    assert "владелец" in result.content.lower()


async def test_embedding_failure_becomes_tool_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    rag = make_rag_service(tmp_path, logger, ExplodingEmbeddings(EmbeddingError("down")))
    tool = make_tool_with(rag)

    result = await tool.execute(
        REQUEST_ID, search_call("сколько дней отпуска"), PROGRESS, CONTEXT_A
    )

    # Сбой инфраструктуры не рвёт цикл: модель получает честный результат.
    assert result.succeeded is False
    assert "недоступен" in result.content.lower()


async def test_query_contents_are_not_logged(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
) -> None:
    """Содержимое запроса в логах не появляется (политика сокрытия содержимого)."""
    secret_query = "зарплата директора по имени Иван"
    rag = make_rag_service(tmp_path, logger)
    await rag.index_document(REQUEST_ID, OWNER_A, "reglament.txt", DOCUMENT_A.encode())

    await make_tool_with(rag).execute(REQUEST_ID, search_call(secret_query), PROGRESS, CONTEXT_A)

    for call in capturing_logger.calls:
        assert "Иван" not in json.dumps(call.kwargs, ensure_ascii=False, default=str)
