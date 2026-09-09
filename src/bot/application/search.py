"""Инструмент агента search_documents: поиск по корпусу владельца.

Реализует Protocol ``Tool`` структурно (как ``MeteredTool`` в метриках):
агентному циклу важен только контракт, сборка — в композиционном корне.
Скоуп владельца приходит контекстом выполнения (ADR-0002): поиск идёт
только по корпусу этого владельца, чужие чанки недостижимы. Перед поиском
запрос переписывается в самостоятельный через порт rag-слоя ``QueryRewriter``
(вопрос с местоимениями сам по себе чанков не находит); сбой переписывания
переводит поиск на сырой запрос. Модель получает чанки с Источниками либо
явное «ничего не найдено»: пустой результат ниже порога близости — данные
для следующего шага, а не сбой. Сбой инфраструктуры (модель переписывания,
эмбеддинг-модель, rag-БД) тоже не рвёт цикл. Содержимое запросов, реплик
и чанков в метрики и логи не попадает.
"""

import json

import structlog

from bot.agent.progress import AgentProgress
from bot.application.errors import EmbeddingError, RagError
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ExecutionContext, ToolCall, ToolParameter, ToolResult, ToolSpec
from bot.rag.models import SearchHit
from bot.rag.rewrite import QueryRewriter
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
    """Поиск top-K чанков корпуса владельца по Поисковому запросу.

    Перед поиском запрос переписывается в самостоятельный: уточнение
    с местоимениями («А можно перенести их на следующий год?») без этого
    не находит чанки про отпуск.
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        rag: RagService,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._rewriter = rewriter
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
        search_query = await self._rewrite(request_id, query, context)
        try:
            hits = await self._rag.search(request_id, context.owner_id, search_query)
        except (EmbeddingError, RagError):
            self._logger.warning("search_failed", request_id=request_id, status="error")
            return ToolResult(content=_UNAVAILABLE_TEXT, succeeded=False)
        if not hits:
            return ToolResult(content=_NOT_FOUND_TEXT, succeeded=True)
        return ToolResult(content=_format_hits(hits), succeeded=True)

    async def _rewrite(
        self,
        request_id: RequestId,
        query: str,
        context: ExecutionContext,
    ) -> str:
        """Переписать запрос в самостоятельный; сбой — поиск по сырому запросу.

        Переписывание выполняется всегда: один короткий вызов модели дешевле
        и предсказуемее эвристик «похож ли вопрос на уточнение». Реплики
        диалога приходят контекстом выполнения; в лог идёт только
        идентификатор запуска, не содержимое запроса и реплик.
        """
        try:
            return await self._rewriter.rewrite(request_id, query, context.recent_turns)
        except Exception:
            self._logger.warning("query_rewrite_failed", request_id=request_id)
            return query


__all__ = ["SearchDocumentsTool"]
