"""Инструмент агента search_documents: поиск по корпусу владельца.

Реализует Protocol ``Tool`` структурно (как ``MeteredTool`` в метриках):
агентному циклу важен только контракт, сборка — в композиционном корне.
Скоуп владельца приходит контекстом выполнения (ADR-0002): поиск идёт
только по корпусу этого владельца, чужие чанки недостижимы. Модель получает
чанки с Источниками либо явное «ничего не найдено»: пустой результат ниже
порога близости — данные для следующего шага, а не сбой. Сбой инфраструктуры
(эмбеддинг-модель, rag-БД) тоже превращается в результат инструмента — цикл
не рвётся. Содержимое запроса и чанков в метрики и логи не попадает.
"""

import json

import structlog

from bot.agent.progress import AgentProgress
from bot.application.errors import EmbeddingError, RagError
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ExecutionContext, ToolCall, ToolParameter, ToolResult, ToolSpec
from bot.rag.models import SearchHit
from bot.rag.service import RagService

_NOT_FOUND_TEXT = (
    "По данному запросу в документах ничего не найдено. "
    "Сообщи пользователю, что ответа в его документах нет."
)
_NO_OWNER_TEXT = "Поиск по документам недоступен: владелец не определён."
_UNAVAILABLE_TEXT = "Поиск по документам сейчас недоступен. Ответь пользователю без поиска."


def _query_from_arguments(arguments: str) -> str | None:
    """Извлечь запрос из JSON-аргументов вызова; None — аргументы невалидны."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    return query


def _format_hits(hits: tuple[SearchHit, ...]) -> str:
    """Чанки с Источниками одним текстом для tool-сообщения модели."""
    blocks = []
    for hit in hits:
        source = f"Источник: {hit.document_name}"
        if hit.page is not None:
            source += f", страница {hit.page}"
        source += f", фрагмент {hit.position}"
        blocks.append(f"{source}\n{hit.text}")
    return f"Найдено фрагментов: {len(hits)}\n\n" + "\n\n".join(blocks)


class SearchDocumentsTool:
    """Поиск top-K чанков корпуса владельца по Поисковому запросу."""

    def __init__(self, rag: RagService, logger: structlog.stdlib.BoundLogger) -> None:
        self._rag = rag
        self._logger = logger.bind(component="search_tool")
        self._spec = ToolSpec(
            name=ToolId("search_documents"),
            description=(
                "Искать по документам пользователя: возвращает найденные фрагменты "
                "с Источниками. Запрос формулируй самостоятельно, без ссылок на "
                "контекст разговора."
            ),
            parameters=(
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Поисковый запрос: самостоятельная формулировка вопроса "
                        "без ссылок на контекст разговора"
                    ),
                ),
            ),
            required=("query",),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(
        self,
        request_id: RequestId,
        call: ToolCall,
        progress: AgentProgress,
        context: ExecutionContext,
    ) -> ToolResult:
        """Найти чанки корпуса владельца; любые неудачи — результат для модели."""
        del progress  # поиск быстрый и тихий: шагов в чате не порождает
        query = _query_from_arguments(call.arguments)
        if query is None:
            return ToolResult(
                content=(
                    "invalid tool arguments: expected JSON object with a string field "
                    f'"query", got: {call.arguments}'
                ),
                succeeded=False,
            )
        if context.owner_id is None:
            return ToolResult(content=_NO_OWNER_TEXT, succeeded=False)
        try:
            hits = await self._rag.search(request_id, context.owner_id, query)
        except (EmbeddingError, RagError):
            self._logger.warning("search_failed", request_id=request_id, status="error")
            return ToolResult(content=_UNAVAILABLE_TEXT, succeeded=False)
        if not hits:
            return ToolResult(content=_NOT_FOUND_TEXT, succeeded=True)
        return ToolResult(content=_format_hits(hits), succeeded=True)


__all__ = ["SearchDocumentsTool"]
