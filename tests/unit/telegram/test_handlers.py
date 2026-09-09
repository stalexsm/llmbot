"""Unit tests for aiogram handlers; Telegram API calls are intercepted."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.methods import SendChatAction, SendMessage
from pytest import MonkeyPatch

import bot.telegram.handlers as handlers_module
from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.application.documents import DocumentService
from bot.application.errors import InferenceTimeoutError, InferenceUnavailableError
from bot.application.models import UserMessageResponse
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId, RequestId, TelegramChatId
from bot.inference.provider import InferenceProvider
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers
from bot.telegram.loader import DocumentLoader
from tests.fakes import (
    FailingInferenceProvider,
    MockInferenceProvider,
    ScriptedInferenceProvider,
    SpyMetricsCollector,
    exec_call_response,
    final_response,
    make_telegram_message,
)


def make_session_store(directory: Path, logger: structlog.stdlib.BoundLogger) -> ChatSessionStore:
    """Реальный SQLite-store над мигрированной временной БД."""
    database = directory / "chats.db"
    apply_migrations(database)
    return ChatSessionStore(database=database, logger=logger)


def make_service(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    tmp_path: Path,
    *,
    step_limit: int = 10,
    with_exec_tool: bool = False,
) -> ApplicationService:
    tools: tuple[ExecTool, ...] = ()
    if with_exec_tool:
        tools = (ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger),)
    loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt="Ты тестовый агент.",
        tools=tools,
        step_limit=step_limit,
        logger=logger,
    )
    return ApplicationService(
        agent=loop,
        sessions=make_session_store(tmp_path, logger),
        history_limit=20,
        logger=logger,
        metrics=SpyMetricsCollector(),
    )


def make_handlers(
    logger: structlog.stdlib.BoundLogger, service: ApplicationService
) -> TelegramHandlers:
    return TelegramHandlers(
        service=service,
        documents=AsyncMock(spec=DocumentService),
        document_loader=AsyncMock(spec=DocumentLoader),
        logger=logger,
        allowed_chat_ids=frozenset(),
    )


def make_stub_handlers(
    logger: structlog.stdlib.BoundLogger,
    allowed_chat_ids: frozenset[TelegramChatId],
    *,
    response_text: str = "Ответ модели",
) -> tuple[TelegramHandlers, AsyncMock]:
    """Хендлеры с полностью подменённым сервисом: проверяется только Telegram-слой."""
    service = AsyncMock(spec=ApplicationService)
    service.process_message.return_value = UserMessageResponse(
        request_id=RequestId("stub-request"), text=response_text
    )
    handlers = TelegramHandlers(
        service=service,
        documents=AsyncMock(spec=DocumentService),
        document_loader=AsyncMock(spec=DocumentLoader),
        logger=logger,
        allowed_chat_ids=allowed_chat_ids,
    )
    return handlers, service


def mock_telegram_api(bot: Bot, monkeypatch: MonkeyPatch) -> AsyncMock:
    """Intercept outgoing Telegram API calls without any network I/O."""
    request_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bot.session, "make_request", request_mock)
    return request_mock


def sent_message(request_mock: AsyncMock) -> SendMessage:
    """Extract the single outgoing Telegram API call from the mock."""
    assert request_mock.await_args is not None
    sent = request_mock.await_args.args[1]
    assert isinstance(sent, SendMessage)
    return sent


async def test_start_answers_with_greeting(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    handlers = make_handlers(logger, make_service(logger, MockInferenceProvider(), tmp_path))
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("/start").as_(bot)
    await handlers.handle_start(message)

    assert sent_message(request_mock).text == handlers_module._START_TEXT


async def test_new_command_resets_session_and_confirms(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path)
    handlers = make_handlers(logger, service)
    request_mock = mock_telegram_api(bot, monkeypatch)
    # В чате уже есть история: она должна быть сброшена командой /new.
    await handlers.handle_text(make_telegram_message("Меня зовут Саша").as_(bot))
    request_mock.reset_mock()

    await handlers.handle_new(make_telegram_message("/new").as_(bot))

    assert sent_message(request_mock).text == handlers_module._NEW_CHAT_TEXT
    sessions = make_session_store(tmp_path, logger)
    assert sessions.load(TelegramChatId(100)) == ()
    # Следующее сообщение не видит сброшенной истории (только системный промпт + вопрос).
    await handlers.handle_text(make_telegram_message("Как меня зовут?").as_(bot))
    assert len(provider.requests[1].messages) == 2


async def test_text_message_returns_model_response(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    provider = MockInferenceProvider(response_content="Ответ модели")
    handlers = make_handlers(logger, make_service(logger, provider, tmp_path))
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    assert sent_message(request_mock).text == "Ответ модели"
    assert len(provider.requests) == 1


def _typing_actions(request_mock: AsyncMock) -> list[SendChatAction]:
    """Все chat action вызовы, перехваченные моком сессии."""
    return [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendChatAction)
    ]


async def test_text_message_shows_typing_while_processing(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    handlers = make_handlers(
        logger,
        make_service(logger, MockInferenceProvider(response_content="Ответ модели"), tmp_path),
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    actions = _typing_actions(request_mock)
    assert actions, "при получении запроса должен отправляться chat action"
    assert all(action.action == ChatAction.TYPING for action in actions)
    calls = [call.args[1] for call in request_mock.await_args_list]
    assert calls.index(actions[0]) < calls.index(sent_message(request_mock))


async def test_typing_action_sent_even_when_inference_fails(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    handlers = make_handlers(
        logger,
        make_service(logger, FailingInferenceProvider(InferenceUnavailableError("down")), tmp_path),
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    assert _typing_actions(request_mock)
    assert sent_message(request_mock).text == handlers_module._ERROR_TEXT


@pytest.mark.parametrize(
    "error",
    [InferenceUnavailableError("down"), InferenceTimeoutError("slow")],
)
async def test_application_error_converted_to_safe_message(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    error: Exception,
) -> None:
    handlers = make_handlers(
        logger, make_service(logger, FailingInferenceProvider(error), tmp_path)
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    assert sent_message(request_mock).text == handlers_module._ERROR_TEXT


async def test_router_routes_new_command_to_reset(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    from aiogram import Dispatcher
    from aiogram.types import Update

    handlers = make_handlers(logger, make_service(logger, MockInferenceProvider(), tmp_path))
    dispatcher = Dispatcher()
    handlers.register(dispatcher)
    request_mock = mock_telegram_api(bot, monkeypatch)

    update = Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1735689600,
                "chat": {"id": 100, "type": "private"},
                "from": {"id": 7, "is_bot": False, "first_name": "Tester"},
                "text": "/new",
            },
        }
    )
    await dispatcher.feed_update(bot, update)

    assert sent_message(request_mock).text == handlers_module._NEW_CHAT_TEXT


async def test_new_command_failure_converted_to_safe_message(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bot.application.errors import InferenceUnavailableError

    service = make_service(logger, MockInferenceProvider(), tmp_path)
    monkeypatch.setattr(
        service, "reset_session", AsyncMock(side_effect=InferenceUnavailableError("down"))
    )
    handlers = make_handlers(logger, service)
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_new(make_telegram_message("/new").as_(bot))  # must not raise

    assert sent_message(request_mock).text == handlers_module._ERROR_TEXT


async def test_command_execution_sends_only_final_answer(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Выполняемые команды — внутренняя кухня: в чате только финальный ответ."""
    # Модель просит exec, видит результат, отвечает финальным ответом.
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo привет"), final_response("Итог: привет")]
    )
    handlers = make_handlers(logger, make_service(logger, provider, tmp_path, with_exec_tool=True))
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("покажи привет").as_(bot))

    calls = [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendMessage)
    ]
    assert [call.text for call in calls] == ["Итог: привет"]


async def test_step_limit_receives_honest_stop_message(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = ScriptedInferenceProvider([exec_call_response("true"), exec_call_response("true")])
    handlers = make_handlers(
        logger, make_service(logger, provider, tmp_path, step_limit=2, with_exec_tool=True)
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("зациклись").as_(bot))

    calls = [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendMessage)
    ]
    # Прогресс шагов в чат не выводится: единственное сообщение — честная остановка.
    assert [call.text for call in calls] == [handlers_module._STEP_LIMIT_TEXT]


# --- Allowlist чатов ---------------------------------------------------------


async def test_empty_allowlist_answers_any_chat(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, service = make_stub_handlers(logger, allowed_chat_ids=frozenset())
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Привет", chat_id=999).as_(bot))

    service.process_message.assert_awaited_once()
    assert sent_message(request_mock).text == "Ответ модели"


async def test_filled_allowlist_ignores_foreign_chat_silently(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, service = make_stub_handlers(
        logger, allowed_chat_ids=frozenset({TelegramChatId(100)})
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Привет", chat_id=999).as_(bot))

    service.process_message.assert_not_awaited()
    assert request_mock.await_count == 0


async def test_filled_allowlist_ignores_foreign_commands_silently(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, service = make_stub_handlers(
        logger, allowed_chat_ids=frozenset({TelegramChatId(100)})
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_start(make_telegram_message("/start", chat_id=999).as_(bot))
    await handlers.handle_new(make_telegram_message("/new", chat_id=999).as_(bot))

    service.reset_session.assert_not_awaited()
    assert request_mock.await_count == 0


async def test_filled_allowlist_listed_chat_works_as_before(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, service = make_stub_handlers(
        logger, allowed_chat_ids=frozenset({TelegramChatId(100)})
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Привет").as_(bot))
    await handlers.handle_new(make_telegram_message("/new").as_(bot))

    service.process_message.assert_awaited_once()
    service.reset_session.assert_awaited_once_with(TelegramChatId(100))
    sent = [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendMessage)
    ]
    assert [call.text for call in sent] == [
        "Ответ модели",
        handlers_module._NEW_CHAT_TEXT,
    ]


async def test_long_reply_arrives_as_ordered_parts(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    long_text = "строка ответа\n" * 600  # 8400 символов: строки по 14
    handlers, _service = make_stub_handlers(
        logger, allowed_chat_ids=frozenset(), response_text=long_text
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Покажи много текста").as_(bot))

    sent = [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendMessage)
    ]
    # Разрезы приходятся на границы строк: 285 строк по 14 символов = 3990 в части.
    assert [len(call.text) for call in sent] == [3990, 3990, 420]
    assert "".join(call.text for call in sent) == long_text
