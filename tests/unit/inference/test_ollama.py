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
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
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
