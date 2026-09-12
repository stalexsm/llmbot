"""E2E: вопрос пользователя → вызов search_documents → ответ с Источником.

Скриптованный провайдер инференса (без сети): модель просит поиск, получает
чанки корпуса владельца и отвечает с Источником; вопрос мимо корпуса —
честное «не найдено». Остальной граф настоящий: AgentLoop, ApplicationService,
DocumentService, RagService с фейковыми эмбеддингами, SQLite, метрики.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import structlog.stdlib
from aiogram import Bot
from aiogram.methods import SendMessage
from pytest import MonkeyPatch
from structlog.testing import CapturingLogger

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import SYSTEM_PROMPT
from bot.application.documents import DocumentService, DocumentUpload
from bot.application.errors import InferenceUnavailableError
from bot.application.search import SearchDocumentsTool
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.domain.messages import MessageRole
from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider
from bot.main import LlmQueryRewriter
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.recorder import MetricsRecorder
from bot.metrics.tool import MeteredTool
from bot.rag.service import RagService
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from bot.telegram.formatter import escape_markdown_v2
from bot.telegram.handlers import TelegramHandlers
from bot.telegram.loader import DocumentLoader
from tests.fakes import (
    KeywordEmbeddingProvider,
    ScriptedInferenceProvider,
    StubQueryRewriter,
    final_response,
    make_rag_service,
    make_telegram_message,
    mock_telegram_api,
    tool_call_response,
)

_OWNER_A = TelegramUserId(7)
_OWNER_B = TelegramUserId(8)

# Ключевые слова тестового корпуса: «отпуск» и «командировка» — оси эмбеддингов;
# тексты без них (сырое уточнение с местоимением) ни с чем не совпадают.
_KEYWORDS = ("отпуск", "командировка")

_DOCUMENT = (
    "Регламент отпусков\n"
    "\n"
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. Отпуск можно\n"
    "разделить на части, если хотя бы одна из них не короче 14 дней.\n"
)


class FailNthCallProvider:
    """Обёртка провайдера: вызов с заданным номером падает, остальные — по сценарию."""

    def __init__(self, inner: InferenceProvider, fail_index: int, error: Exception) -> None:
        self._inner = inner
        self._fail_index = fail_index
        self._error = error
        self.calls = 0

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        if self.calls == self._fail_index:
            raise self._error
        return await self._inner.generate(request)


def make_stack(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    tmp_path: Path,
    *,
    rewriter: StubQueryRewriter | None = None,
    rag: RagService | None = None,
) -> tuple[TelegramHandlers, DocumentService]:
    """Полный граф сообщения: скриптованный инференс + настоящий RAG.

    Переписывание запроса по умолчанию — LlmQueryRewriter над тем же
    провайдером (как в композиционном корне); для сценариев без живого
    переписывания подаётся фейковка порта.
    """
    metrics_collector = RunMetricsCollector(
        recorder=MetricsRecorder(directory=tmp_path / "metrics", logger=logger),
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
    )
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    rag = rag if rag is not None else make_rag_service(tmp_path, logger)
    documents = DocumentService(rag, logger)
    query_rewriter = (
        rewriter
        if rewriter is not None
        else LlmQueryRewriter(provider, ModelId("qwen3:1.7b"), logger)
    )
    agent_loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt=SYSTEM_PROMPT,
        tools=(
            MeteredTool(inner=exec_tool, collector=metrics_collector),
            MeteredTool(
                inner=SearchDocumentsTool(query_rewriter, rag, logger),
                collector=metrics_collector,
            ),
        ),
        step_limit=10,
        logger=logger,
    )
    chat_database = tmp_path / "chats.db"
    apply_migrations(chat_database)
    service = ApplicationService(
        agent=agent_loop,
        sessions=ChatSessionStore(database=chat_database, logger=logger),
        history_limit=20,
        logger=logger,
        metrics=metrics_collector,
    )
    handlers = TelegramHandlers(
        service=service,
        documents=documents,
        document_loader=AsyncMock(spec=DocumentLoader),  # загрузки файлов в тесте не участвуют
        logger=logger,
        allowed_chat_ids=frozenset(),
    )
    return handlers, documents


def sent_reply(request_mock: AsyncMock) -> SendMessage:
    sent = request_mock.await_args_list[-1].args[1]
    assert isinstance(sent, SendMessage)
    return sent


def tool_call_events(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "metrics" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if json.loads(line)["kind"] == "tool_call"]


async def test_question_routes_to_search_and_answer_cites_source(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Вопрос по документу → search_documents → финальный ответ с Источником."""
    provider = ScriptedInferenceProvider(
        [
            tool_call_response(
                "search_documents",
                json.dumps({"query": "сколько дней отпуска"}, ensure_ascii=False),
            ),
            final_response("сколько дней отпуска"),
            final_response("Ежегодный отпуск — 28 календарных дней.\n\nИсточник: reglament.txt"),
        ]
    )
    handlers, documents = make_stack(logger, provider, tmp_path)
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Сколько дней отпуска?").as_(bot))

    # Модель получила спеку поиска, вызвала его и увидела чанк с Источником.
    first_tools = [str(spec.name) for spec in provider.requests[0].tools]
    assert "search_documents" in first_tools
    tool_message = provider.requests[2].messages[3]
    assert tool_message.role is MessageRole.TOOL
    assert "Источник: reglament.txt" in tool_message.content
    assert "28 календарных дней" in tool_message.content
    # Пользователь получает ответ с Источником.
    assert sent_reply(request_mock).text == escape_markdown_v2(
        "Ежегодный отпуск — 28 календарных дней.\n\nИсточник: reglament.txt"
    )
    # Метрики: поиск записан событием tool_call без содержимого запроса.
    events = tool_call_events(tmp_path)
    assert [event["tool_name"] for event in events] == ["search_documents"]
    assert events[0]["succeeded"] is True
    assert "сколько дней отпуска" not in json.dumps(events[0], ensure_ascii=False)
    assert "28 календарных" not in json.dumps(events[0], ensure_ascii=False)


async def test_off_corpus_question_gets_honest_not_found(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = ScriptedInferenceProvider(
        [
            tool_call_response(
                "search_documents",
                json.dumps({"query": "расписание поездов до Владивостока"}, ensure_ascii=False),
            ),
            final_response("расписание поездов до Владивостока"),
            final_response("Я не нашёл ответа на этот вопрос в ваших документах."),
        ]
    )
    handlers, documents = make_stack(logger, provider, tmp_path)
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Когда отходит поезд?").as_(bot))

    # Инструмент вернул явное «не найдено», модель ответила честно.
    tool_message = provider.requests[2].messages[3]
    assert "ничего не найдено" in tool_message.content.lower()
    assert sent_reply(request_mock).text == escape_markdown_v2(
        "Я не нашёл ответа на этот вопрос в ваших документах."
    )


async def test_second_owner_never_sees_first_owner_chunks(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Изоляция сквозь весь стек: владелец B ищет то же слово — чанков A нет."""
    provider = ScriptedInferenceProvider(
        [
            tool_call_response(
                "search_documents",
                json.dumps({"query": "сколько дней отпуска"}, ensure_ascii=False),
            ),
            final_response("сколько дней отпуска"),
            final_response("Я не нашёл ответа на этот вопрос в ваших документах."),
        ]
    )
    handlers, documents = make_stack(logger, provider, tmp_path)
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(
        make_telegram_message("Сколько дней отпуска?", user_id=int(_OWNER_B)).as_(bot)
    )

    # Поиск владельца B не вернул чанки A: изоляция скоупа владельца.
    tool_message = provider.requests[2].messages[3]
    assert "ничего не найдено" in tool_message.content.lower()
    assert "28 календарных дней" not in tool_message.content
    assert sent_reply(request_mock).text == escape_markdown_v2(
        "Я не нашёл ответа на этот вопрос в ваших документах."
    )


async def test_pronoun_followup_finds_chunks_after_rewrite(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Вопрос → уточнение с местоимением: переписанный запрос находит чанки.

    Сырой уточняющий вопрос «перенести их на следующий год» не содержит
    ни одного ключевого слова корпуса и ниже порога близости; после
    переписывания («перенос отпуска на следующий год») чанк находится.
    """
    provider = ScriptedInferenceProvider(
        [
            # Сообщение 1: самостоятельный вопрос.
            tool_call_response(
                "search_documents",
                json.dumps({"query": "сколько дней отпуска"}, ensure_ascii=False),
            ),
            final_response("сколько дней отпуска"),
            final_response("Ежегодный отпуск — 28 календарных дней.\n\nИсточник: reglament.txt"),
            # Сообщение 2: уточнение с местоимением.
            tool_call_response(
                "search_documents",
                json.dumps({"query": "перенести их на следующий год"}, ensure_ascii=False),
            ),
            final_response("перенос отпуска на следующий год"),
            final_response(
                "Перенести отпуск на следующий год можно, разделив его на части."
                "\n\nИсточник: reglament.txt"
            ),
        ]
    )
    embeddings = KeywordEmbeddingProvider(_KEYWORDS)
    rag = make_rag_service(tmp_path, logger, embeddings=embeddings)
    handlers, documents = make_stack(logger, provider, tmp_path, rag=rag)
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Сколько дней отпуска?").as_(bot))
    await handlers.handle_text(
        make_telegram_message("А можно перенести их на следующий год?").as_(bot)
    )

    # Переписывание получило реплики диалога (включая текущий вопрос)
    # и сырой запрос модели; без инструментов, с id запуска.
    rewrite_request = provider.requests[4]
    assert rewrite_request.tools == ()
    rewrite_prompt = rewrite_request.messages[1].content
    assert "Пользователь: Сколько дней отпуска?" in rewrite_prompt
    assert "Ассистент: Ежегодный отпуск — 28 календарных дней." in rewrite_prompt
    assert "Пользователь: А можно перенести их на следующий год?" in rewrite_prompt
    assert "перенести их на следующий год" in rewrite_prompt
    # В эмбеддинги ушёл переписанный запрос, а не сырой уточняющий.
    assert [request.texts for request in embeddings.requests[1:]] == [
        ("сколько дней отпуска",),
        ("перенос отпуска на следующий год",),
    ]
    # Модель увидела чанк, который сырой уточняющий вопрос не нашёл бы.
    tool_message = provider.requests[5].messages[-1]
    assert tool_message.role is MessageRole.TOOL
    assert "28 календарных дней" in tool_message.content
    assert "Источник: reglament.txt" in tool_message.content
    # Пользователь получил ответ с Источником; вызовы поиска — в метриках.
    assert escape_markdown_v2("Источник: reglament.txt") in sent_reply(request_mock).text
    assert [event["tool_name"] for event in tool_call_events(tmp_path)] == [
        "search_documents",
        "search_documents",
    ]


async def test_raw_pronoun_query_without_rewrite_finds_nothing(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Без переписывания тот же сырой уточняющий вопрос чанков не находит."""
    provider = ScriptedInferenceProvider(
        [
            tool_call_response(
                "search_documents",
                json.dumps({"query": "перенести их на следующий год"}, ensure_ascii=False),
            ),
            final_response("Я не нашёл ответа на этот вопрос в ваших документах."),
        ]
    )
    handlers, documents = make_stack(logger, provider, tmp_path, rewriter=StubQueryRewriter())
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(
        make_telegram_message("А можно перенести их на следующий год?").as_(bot)
    )

    # Пасsthrough-переписывание: сырой запрос ушёл в поиск как есть — мимо корпуса.
    tool_message = provider.requests[1].messages[3]
    assert "ничего не найдено" in tool_message.content.lower()
    assert sent_reply(request_mock).text == escape_markdown_v2(
        "Я не нашёл ответа на этот вопрос в ваших документах."
    )


async def test_rewrite_failure_searches_raw_and_user_gets_answer(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capturing_logger: CapturingLogger,
) -> None:
    """Сбой переписывания не рвёт поиск: сырой вопрос, пользователь с ответом."""
    scripted = ScriptedInferenceProvider(
        [
            tool_call_response(
                "search_documents",
                json.dumps({"query": "сколько дней отпуска"}, ensure_ascii=False),
            ),
            final_response("Ежегодный отпуск — 28 календарных дней.\n\nИсточник: reglament.txt"),
        ]
    )
    # Порядок вызовов: шаг цикла → переписывание (падает) → шаг цикла.
    provider = FailNthCallProvider(
        scripted, fail_index=2, error=InferenceUnavailableError("rewrite down")
    )
    handlers, documents = make_stack(logger, provider, tmp_path)
    await documents.index(
        DocumentUpload(
            request_id=RequestId(str(uuid4())),
            owner_id=_OWNER_A,
            name="reglament.txt",
            content=_DOCUMENT.encode(),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Сколько дней отпуска?").as_(bot))

    # Первый вызов провайдера (переписывание) упал, поиск пошёл по сырому
    # запросу и нашёл чанк; пользователь ответ получил.
    tool_message = scripted.requests[1].messages[3]
    assert "28 календарных дней" in tool_message.content
    assert escape_markdown_v2("Источник: reglament.txt") in sent_reply(request_mock).text
    # Сбой переписывания отмечен в логе без содержимого запроса.
    assert any(
        call.kwargs.get("event") == "query_rewrite_failed" for call in capturing_logger.calls
    )
    for call in capturing_logger.calls:
        assert "отпуск" not in json.dumps(call.kwargs, ensure_ascii=False, default=str)
