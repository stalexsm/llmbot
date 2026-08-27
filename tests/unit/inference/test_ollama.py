"""Unit tests for the Ollama adapter. No running Ollama instance is required."""

import json
from collections.abc import Callable, Coroutine
from uuid import uuid4

import httpx
import pytest
import structlog.stdlib
from httpx import MockTransport

from bot.application.errors import (
    InferenceError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall, ToolParameter, ToolSpec
from bot.inference.models import InferenceRequest
from bot.inference.ollama import OllamaInferenceProvider

TransportHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def make_request() -> InferenceRequest:
    return InferenceRequest(
        request_id=RequestId(str(uuid4())),
        model=ModelId("qwen3:1.7b"),
        messages=(InferenceMessage(role=MessageRole.USER, content="Привет"),),
    )


def make_provider(
    logger: structlog.stdlib.BoundLogger,
    handler: TransportHandler,
) -> tuple[httpx.AsyncClient, OllamaInferenceProvider]:
    client = httpx.AsyncClient(transport=MockTransport(handler))
    provider = OllamaInferenceProvider(
        client=client,
        base_url="http://ollama.test",
        timeout_seconds=0.5,
        logger=logger,
    )
    return client, provider


async def ok_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode("utf-8"))
    assert body["stream"] is False
    assert body["model"] == "qwen3:1.7b"
    assert body["messages"] == [{"role": "user", "content": "Привет"}]
    return httpx.Response(
        200,
        json={
            "model": "qwen3:1.7b",
            "created_at": "2025-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Ответ модели"},
            "done": True,
        },
    )


async def test_successful_inference(logger: structlog.stdlib.BoundLogger) -> None:
    client, provider = make_provider(logger, ok_handler)

    async with client:
        response = await provider.generate(make_request())

    assert response.content == "Ответ модели"


async def test_think_is_false_by_default(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["think"] is False
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Ответ"}})

    client, provider = make_provider(logger, handler)

    async with client:
        await provider.generate(make_request())


async def test_think_true_reaches_the_wire(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["think"] is True
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Ответ"}})

    client = httpx.AsyncClient(transport=MockTransport(handler))
    provider = OllamaInferenceProvider(
        client=client,
        base_url="http://ollama.test",
        timeout_seconds=0.5,
        logger=logger,
        think=True,
    )

    async with client:
        await provider.generate(make_request())


async def test_connect_error_maps_to_unavailable(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(InferenceUnavailableError):
            await provider.generate(make_request())


async def test_timeout_maps_to_timeout_error(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(InferenceTimeoutError):
            await provider.generate(make_request())


@pytest.mark.parametrize("status_code", [404, 500, 503])
async def test_http_error_status_maps_to_unavailable(
    logger: structlog.stdlib.BoundLogger, status_code: int
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(InferenceUnavailableError):
            await provider.generate(make_request())


@pytest.mark.parametrize(
    ("payload", "body"),
    [
        ({"message": {"role": "assistant", "content": 123}}, None),
        ({"message": None}, None),
        ({"unexpected": True}, None),
        (None, "not-json-at-all"),
        (None, ""),
    ],
)
async def test_malformed_response_maps_to_inference_error(
    logger: structlog.stdlib.BoundLogger,
    payload: dict[str, object] | None,
    body: str | None,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if body is not None:
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=payload)

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(InferenceError):
            await provider.generate(make_request())


async def test_empty_content_is_valid_at_adapter_level(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": ""}, "done": True},
        )

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.generate(make_request())

    assert response.content == ""


EXEC_SPEC = ToolSpec(
    name=ToolId("exec"),
    description="Выполнить консольную команду",
    parameters=(ToolParameter(name="command", type="string", description="Команда"),),
    required=("command",),
)


def make_tooled_request() -> InferenceRequest:
    return InferenceRequest(
        request_id=RequestId(str(uuid4())),
        model=ModelId("qwen3:1.7b"),
        messages=(
            InferenceMessage(role=MessageRole.USER, content="покажи файлы"),
            InferenceMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name=ToolId("exec"), arguments='{"command": "ls"}'),),
            ),
            InferenceMessage(
                role=MessageRole.TOOL,
                content="exit_code: 0",
                tool_name=ToolId("exec"),
            ),
        ),
        tools=(EXEC_SPEC,),
    )


async def test_tools_and_think_are_serialized(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    captured: dict[str, dict] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        captured["body"] = json_module.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Готово"}, "done": True},
        )

    client, provider = make_provider(logger, handler)

    async with client:
        await provider.generate(make_tooled_request())

    body = captured["body"]
    # Режим размышлений модели отключён на уровне провайдера.
    assert body["think"] is False
    assert body["stream"] is False
    tools = body["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    function = tools[0]["function"]
    assert function["name"] == "exec"
    assert function["description"] == "Выполнить консольную команду"
    assert function["parameters"]["type"] == "object"
    assert function["parameters"]["required"] == ["command"]
    assert function["parameters"]["properties"]["command"] == {
        "type": "string",
        "description": "Команда",
    }


async def test_history_with_tool_pairs_is_serialized(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    captured: dict[str, dict] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        captured["body"] = json_module.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Готово"}, "done": True},
        )

    client, provider = make_provider(logger, handler)

    async with client:
        await provider.generate(make_tooled_request())

    messages = captured["body"]["messages"]
    assert messages[0] == {"role": "user", "content": "покажи файлы"}
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "exec", "arguments": {"command": "ls"}}},
        ],
    }
    assert messages[2] == {
        "role": "tool",
        "content": "exit_code: 0",
        "tool_name": "exec",
    }


async def test_tool_calls_in_response_are_parsed(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
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
                                "arguments": {"command": "ls -la"},
                            }
                        }
                    ],
                },
                "done": True,
            },
        )

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.generate(make_tooled_request())

    assert response.content == ""
    assert response.tool_calls == (
        ToolCall(name=ToolId("exec"), arguments='{"command": "ls -la"}'),
    )


async def test_tool_call_arguments_as_json_string_are_accepted(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    # Старые сборки Ollama возвращали arguments уже строкой — принимаем оба варианта.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec",
                                "arguments": '{"command": "ls"}',
                            }
                        }
                    ],
                },
                "done": True,
            },
        )

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.generate(make_request())

    assert response.tool_calls[0].arguments == '{"command": "ls"}'


async def test_thinking_tags_are_stripped_from_content(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>внутренние размышления</think>Видимый ответ",
                },
                "done": True,
            },
        )

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.generate(make_request())

    assert response.content == "Видимый ответ"


async def test_unclosed_thinking_block_is_stripped(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Ответ<think>незакрытые размышления до конца",
                },
                "done": True,
            },
        )

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.generate(make_request())

    assert response.content == "Ответ"
