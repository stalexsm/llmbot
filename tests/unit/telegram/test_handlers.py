"""Unit tests for aiogram handlers; Telegram API calls are intercepted."""

from unittest.mock import AsyncMock

import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.methods import SendMessage
from pytest import MonkeyPatch

import bot.telegram.handlers as handlers_module
from bot.application.errors import InferenceTimeoutError, InferenceUnavailableError
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId
from bot.inference.provider import InferenceProvider
from bot.telegram.handlers import TelegramHandlers
from tests.fakes import FailingInferenceProvider, MockInferenceProvider, make_telegram_message


def make_service(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
) -> ApplicationService:
    return ApplicationService(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
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
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers = make_handlers(logger, make_service(logger, MockInferenceProvider()))
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("/start").as_(bot)
    await handlers.handle_start(message)

    assert sent_message(request_mock).text == handlers_module._START_TEXT


async def test_text_message_returns_model_response(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    provider = MockInferenceProvider(response_content="Ответ модели")
    handlers = make_handlers(logger, make_service(logger, provider))
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
    error: Exception,
) -> None:
    handlers = make_handlers(logger, make_service(logger, FailingInferenceProvider(error)))
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    assert sent_message(request_mock).text == handlers_module._ERROR_TEXT
