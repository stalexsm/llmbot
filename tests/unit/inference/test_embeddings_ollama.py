"""Unit tests for the Ollama embedding adapter. No running Ollama is required."""

import json
from collections.abc import Callable, Coroutine
from uuid import uuid4

import httpx
import pytest
import structlog.stdlib
from httpx import MockTransport

from bot.application.errors import (
    EmbeddingError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from bot.domain.ids import ModelId, RequestId
from bot.inference.embeddings import EmbeddingRequest
from bot.inference.embeddings_ollama import DEFAULT_BATCH_SIZE, OllamaEmbeddingProvider

TransportHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def make_request(texts: tuple[str, ...]) -> EmbeddingRequest:
    return EmbeddingRequest(
        request_id=RequestId(str(uuid4())),
        model=ModelId("bge-m3"),
        texts=texts,
    )


def make_provider(
    logger: structlog.stdlib.BoundLogger,
    handler: TransportHandler,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[httpx.AsyncClient, OllamaEmbeddingProvider]:
    client = httpx.AsyncClient(transport=MockTransport(handler))
    provider = OllamaEmbeddingProvider(
        client=client,
        base_url="http://ollama.test",
        timeout_seconds=0.5,
        logger=logger,
        batch_size=batch_size,
    )
    return client, provider


def embed_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(200, json={"model": "bge-m3", "embeddings": vectors})


async def test_wire_format_and_parsed_vectors(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return embed_response([[1.0, 0.0], [0.5, 0.5]])

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.embed(make_request(("кот", "погода")))

    assert str(captured["url"]).endswith("/api/embed")
    assert captured["body"] == {"model": "bge-m3", "input": ["кот", "погода"]}
    assert response.embeddings == ((1.0, 0.0), (0.5, 0.5))


async def test_texts_are_sent_in_batches_in_order(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    batches: list[list[str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        batches.append(body["input"])
        # Один и тот же вектор на батч — порядок проверяем по самим батчам.
        return embed_response([[float(len(body["input"])), 0.0]] * len(body["input"]))

    client, provider = make_provider(logger, handler, batch_size=2)

    async with client:
        response = await provider.embed(make_request(("а", "б", "в", "г", "д")))

    assert batches == [["а", "б"], ["в", "г"], ["д"]]
    # Порядок векторов соответствует порядку текстов, а не батчей.
    assert response.embeddings == ((2.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 0.0), (1.0, 0.0))


async def test_empty_request_skips_http_call(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    client, provider = make_provider(logger, handler)

    async with client:
        response = await provider.embed(make_request(()))

    assert response.embeddings == ()


async def test_connect_error_maps_to_unavailable(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(EmbeddingUnavailableError):
            await provider.embed(make_request(("текст",)))


async def test_timeout_maps_to_timeout_error(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(EmbeddingTimeoutError):
            await provider.embed(make_request(("текст",)))


@pytest.mark.parametrize("status_code", [400, 404, 500, 503])
async def test_http_error_status_maps_to_unavailable(
    logger: structlog.stdlib.BoundLogger, status_code: int
) -> None:
    # 404 — типичный ответ Ollama при отсутствии модели эмбеддингов.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(EmbeddingUnavailableError):
            await provider.embed(make_request(("текст",)))


async def test_malformed_response_maps_to_embedding_error(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json-at-all")

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(EmbeddingError):
            await provider.embed(make_request(("текст",)))


async def test_vector_count_mismatch_maps_to_embedding_error(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return embed_response([[1.0, 0.0]])

    client, provider = make_provider(logger, handler)

    async with client:
        with pytest.raises(EmbeddingError):
            await provider.embed(make_request(("раз", "два")))


def test_zero_batch_size_is_rejected(logger: structlog.stdlib.BoundLogger) -> None:
    with pytest.raises(ValueError):
        OllamaEmbeddingProvider(
            client=httpx.AsyncClient(),
            base_url="http://ollama.test",
            timeout_seconds=0.5,
            logger=logger,
            batch_size=0,
        )
