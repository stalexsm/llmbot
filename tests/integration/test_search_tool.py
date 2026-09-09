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

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import SYSTEM_PROMPT
from bot.application.documents import DocumentService, DocumentUpload
from bot.application.search import SearchDocumentsTool
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.domain.messages import MessageRole
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.recorder import MetricsRecorder
from bot.metrics.tool import MeteredTool
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers
from bot.telegram.loader import DocumentLoader
from tests.fakes import (
    ScriptedInferenceProvider,
    final_response,
    make_rag_service,
    make_telegram_message,
    mock_telegram_api,
    tool_call_response,
)

_OWNER_A = TelegramUserId(7)
_OWNER_B = TelegramUserId(8)

_DOCUMENT = (
    "Регламент отпусков\n"
    "\n"
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. Отпуск можно\n"
    "разделить на части, если хотя бы одна из них не короче 14 дней.\n"
)


def make_stack(
    logger: structlog.stdlib.BoundLogger,
    provider: ScriptedInferenceProvider,
    tmp_path: Path,
) -> tuple[TelegramHandlers, DocumentService]:
    """Полный граф сообщения: скриптованный инференс + настоящий RAG."""
    metrics_collector = RunMetricsCollector(
        recorder=MetricsRecorder(directory=tmp_path / "metrics", logger=logger),
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
    )
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    rag = make_rag_service(tmp_path, logger)
    documents = DocumentService(rag, logger)
    agent_loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt=SYSTEM_PROMPT,
        tools=(
            MeteredTool(inner=exec_tool, collector=metrics_collector),
            MeteredTool(inner=SearchDocumentsTool(rag, logger), collector=metrics_collector),
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
    tool_message = provider.requests[1].messages[3]
    assert tool_message.role is MessageRole.TOOL
    assert "Источник: reglament.txt" in tool_message.content
    assert "28 календарных дней" in tool_message.content
    # Пользователь получает ответ с Источником.
    assert (
        sent_reply(request_mock).text
        == "Ежегодный отпуск — 28 календарных дней.\n\nИсточник: reglament.txt"
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
    tool_message = provider.requests[1].messages[3]
    assert "ничего не найдено" in tool_message.content.lower()
    assert sent_reply(request_mock).text == "Я не нашёл ответа на этот вопрос в ваших документах."


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
    tool_message = provider.requests[1].messages[3]
    assert "ничего не найдено" in tool_message.content.lower()
    assert "28 календарных дней" not in tool_message.content
    assert sent_reply(request_mock).text == "Я не нашёл ответа на этот вопрос в ваших документах."
