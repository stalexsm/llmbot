"""Integration tests: full Telegram → Application → Agent → Ollama wiring without network.

The Ollama HTTP API is emulated with ``httpx.MockTransport``; outgoing Telegram
API calls are intercepted; команды exec выполняются настоящим shell в tmp-каталоге.
No Ollama instance or Telegram connection required.
"""

import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.methods import EditMessageText, SendMessage
from httpx import MockTransport
from pytest import MonkeyPatch

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import SYSTEM_PROMPT
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
    *,
    step_limit: int = 10,
) -> TelegramHandlers:
    client = httpx.AsyncClient(transport=MockTransport(handler))
    provider = OllamaInferenceProvider(
        client=client,
        base_url="http://ollama.test",
        timeout_seconds=1.0,
        logger=logger,
    )
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    agent_loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt=SYSTEM_PROMPT,
        tools=(exec_tool,),
        step_limit=step_limit,
        logger=logger,
    )
    service = ApplicationService(
        agent=agent_loop,
        sessions=ChatSessionStore(directory=tmp_path, logger=logger),
        history_limit=20,
        logger=logger,
    )
    return TelegramHandlers(service=service, logger=logger, allowed_chat_ids=frozenset())


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


def progress_mock(bot: Bot, monkeypatch: MonkeyPatch) -> AsyncMock:
    """Мок Telegram API, возвращающий связанное сообщение-статус для правок."""
    status_message = make_telegram_message("статус", message_id=900).as_(bot)
    request_mock = AsyncMock(return_value=status_message)
    monkeypatch.setattr(bot.session, "make_request", request_mock)
    return request_mock


async def test_full_pipeline_text_to_reply(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    handlers = make_stack(logger, ok_handler, tmp_path)
    request_mock = progress_mock(bot, monkeypatch)

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
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ответ"}},
        )

    handlers = make_stack(logger, capturing_handler, tmp_path)
    progress_mock(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("Меня зовут Саша").as_(bot))
    await handlers.handle_text(make_telegram_message("Как меня зовут?").as_(bot))

    assert [(message["role"], message["content"]) for message in bodies[1]["messages"]] == [
        ("system", SYSTEM_PROMPT),
        ("user", "Меня зовут Саша"),
        ("assistant", "Ответ"),
        ("user", "Как меня зовут?"),
    ]


async def test_full_pipeline_exec_tool_roundtrip(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Сквозной агентный сценарий: tool-call → настоящий shell → финальный ответ."""
    bodies: list[dict] = []

    async def scripted_handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:1.7b",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "exec",
                                    "arguments": {"command": "echo привет из shell"},
                                }
                            }
                        ],
                    },
                    "done": True,
                },
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Модель увидела вывод команды"}},
        )

    handlers = make_stack(logger, scripted_handler, tmp_path)
    request_mock = progress_mock(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("выполни команду").as_(bot))

    # Первый запрос: описание инструмента и отключённые размышления.
    first_body = bodies[0]
    assert first_body["tools"][0]["function"]["name"] == "exec"
    assert first_body["think"] is False
    # Второй запрос: пара «вызов → результат» с выводом настоящей команды.
    second_messages = bodies[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2]["tool_calls"][0]["function"]["name"] == "exec"
    tool_message = second_messages[3]
    assert tool_message["tool_name"] == "exec"
    assert "привет из shell" in tool_message["content"]
    # Пользователь увидел шаги и финальный ответ.
    calls = [call.args[1] for call in request_mock.await_args_list]
    assert [type(call) for call in calls] == [SendMessage, EditMessageText, SendMessage]
    assert calls[0].text == "⏳ echo привет из shell"
    assert calls[1].text == "✅ echo привет из shell"
    assert calls[2].text == "Модель увидела вывод команды"


async def test_full_pipeline_step_limit_returns_honest_stop(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def looping_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "exec", "arguments": {"command": "true"}}}
                    ],
                },
                "done": True,
            },
        )

    handlers = make_stack(logger, looping_handler, tmp_path, step_limit=2)
    request_mock = progress_mock(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message("зациклись").as_(bot))

    final_call = request_mock.await_args_list[-1].args[1]
    assert isinstance(final_call, SendMessage)
    assert "лимит шагов" in final_call.text


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
    request_mock = progress_mock(bot, monkeypatch)

    message = make_telegram_message("Привет").as_(bot)
    await handlers.handle_text(message)  # must not raise

    assert sent_message(request_mock).text != "Привет из модели"
