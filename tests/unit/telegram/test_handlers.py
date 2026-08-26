"""Unit tests for aiogram handlers; Telegram API calls are intercepted."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.methods import SendMessage
from pytest import MonkeyPatch

import bot.telegram.handlers as handlers_module
from bot.application.errors import InferenceTimeoutError, InferenceUnavailableError
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId, TelegramChatId
from bot.inference.provider import InferenceProvider
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers
from tests.fakes import FailingInferenceProvider, MockInferenceProvider, make_telegram_message


def make_service(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    tmp_path: Path,
) -> ApplicationService:
    return ApplicationService(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        sessions=ChatSessionStore(directory=tmp_path, logger=logger),
        history_limit=20,
        logger=logger,
    )


def make_handlers(
    logger: structlog.stdlib.BoundLogger, service: ApplicationService
) -> TelegramHandlers:
    return TelegramHandlers(service=service, logger=logger)


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
    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    assert sessions.load(TelegramChatId(100)) == ()
    # Следующее сообщение не видит сброшенной истории.
    await handlers.handle_text(make_telegram_message("Как меня зовут?").as_(bot))
    assert len(provider.requests[1].messages) == 1


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
