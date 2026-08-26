"""Integration tests: full Telegram → Application → Ollama wiring without network.

The Ollama HTTP API is emulated with ``httpx.MockTransport``; outgoing Telegram
API calls are intercepted. No Ollama instance or Telegram connection required.
"""

from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.methods import SendMessage
from httpx import MockTransport
from pytest import MonkeyPatch

from bot.application.service import ApplicationService
from bot.domain.ids import ModelId
from bot.inference.ollama import OllamaInferenceProvider
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers
from tests.fakes import make_telegram_message

TransportHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def make_stack(
    logger: structlog.stdlib.BoundLogger,
    handler: TransportHandler,
    tmp_path: Path,
) -> TelegramHandlers:
    client = httpx.AsyncClient(transport=MockTransport(handler))
    provider = OllamaInferenceProvider(
        client=client,
        base_url="http://ollama.test",
        timeout_seconds=1.0,
        logger=logger,
    )
    service = ApplicationService(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        sessions=ChatSessionStore(directory=tmp_path, logger=logger),
        history_limit=20,
        logger=logger,
    )
    return TelegramHandlers(service=service, logger=logger)


def sent_message(request_mock: AsyncMock) -> SendMessage:
    assert request_mock.await_args is not None
    sent = request_mock.await_args.args[1]
    assert isinstance(sent, SendMessage)
    return sent


async def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": {"role": "assistant", "content": "Привет из модели"}},
    )


async def test_full_pipeline_text_to_reply(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    handlers = make_stack(logger, ok_handler, tmp_path)
    request_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bot.session, "make_request", request_mock)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)

    assert sent_message(request_mock).text == "Привет из модели"


async def test_full_pipeline_second_message_sees_first_exchange(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    bodies: list[dict] = []

    async def capturing_handler(request: httpx.Request) -> httpx.Response:
        import json

        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ответ"}},
        )

    handlers = make_stack(logger, capturing_handler, tmp_path)
    request_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bot.session, "make_request", request_mock)

    await handlers.handle_text(make_telegram_message("Меня зовут Саша").as_(bot))
    await handlers.handle_text(make_telegram_message("Как меня зовут?").as_(bot))

    assert [(message["role"], message["content"]) for message in bodies[1]["messages"]] == [
        ("user", "Меня зовут Саша"),
        ("assistant", "Ответ"),
        ("user", "Как меня зовут?"),
    ]


@pytest.mark.parametrize("status_code", [500, 503])
async def test_full_pipeline_returns_safe_message_when_ollama_down(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    status_code: int,
) -> None:
    async def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    handlers = make_stack(logger, failing_handler, tmp_path)
    request_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bot.session, "make_request", request_mock)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)  # must not raise

    assert sent_message(request_mock).text != "Привет из модели"
