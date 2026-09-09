"""Ollama embedding infrastructure adapter.

Responsibilities:
    - HTTP communication with the Ollama ``/api/embed`` endpoint;
    - batch processing of texts (chunks are split into fixed-size batches);
    - response parsing and validation;
    - timeout handling;
    - mapping infrastructure errors to application-level errors.

The adapter knows nothing about Telegram, chunking, or storage.
"""

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from bot.application.errors import (
    EmbeddingError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from bot.inference.embeddings import EmbeddingRequest, EmbeddingResponse

# Размер батча одного HTTP-вызова: чанки уходят группами, чтобы запрос
# не раздувался до сотен килобайт на больших документах.
DEFAULT_BATCH_SIZE = 32


class _EmbedResponsePayload(BaseModel):
    """Wire format of an Ollama ``/api/embed`` response (external boundary only)."""

    embeddings: list[list[float]]


class OllamaEmbeddingProvider:
    """Embeddings backed by the Ollama ``/api/embed`` endpoint (non-streaming)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_seconds: float,
        logger: structlog.stdlib.BoundLogger,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._embed_url = f"{base_url.rstrip('/')}/api/embed"
        self._timeout = httpx.Timeout(timeout_seconds)
        self._logger = logger.bind(component="ollama_embedding_provider")
        self._batch_size = batch_size

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not request.texts:
            return EmbeddingResponse(request_id=request.request_id, embeddings=())
        embeddings: list[tuple[float, ...]] = []
        for start in range(0, len(request.texts), self._batch_size):
            batch = request.texts[start : start + self._batch_size]
            embeddings.extend(await self._embed_batch(request, batch))
        return EmbeddingResponse(
            request_id=request.request_id,
            embeddings=tuple(embeddings),
        )

    async def _embed_batch(
        self,
        request: EmbeddingRequest,
        batch: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        try:
            http_response = await self._client.post(
                self._embed_url,
                json={"model": request.model, "input": list(batch)},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            self._logger.warning(
                "ollama_embed_timed_out",
                request_id=request.request_id,
                model=request.model,
                batch_size=len(batch),
            )
            raise EmbeddingTimeoutError("Embedding request timed out") from exc
        except httpx.HTTPError as exc:
            self._logger.warning(
                "ollama_embed_failed",
                request_id=request.request_id,
                model=request.model,
                batch_size=len(batch),
            )
            raise EmbeddingUnavailableError("Embedding provider is unreachable") from exc

        if http_response.status_code != httpx.codes.OK:
            self._logger.warning(
                "ollama_embed_http_error",
                request_id=request.request_id,
                model=request.model,
                status_code=http_response.status_code,
            )
            # Отсутствие модели в Ollama (и любой другой сбой на стороне
            # сервера) — прикладная «недоступность», наверх уходит понятная
            # ошибка, а не сырой HTTP-статус.
            raise EmbeddingUnavailableError("Embedding provider returned an error status")

        try:
            payload = _EmbedResponsePayload.model_validate_json(http_response.text)
        except ValidationError as exc:
            self._logger.warning(
                "ollama_embed_invalid_response",
                request_id=request.request_id,
                model=request.model,
            )
            raise EmbeddingError("Embedding provider returned a malformed response") from exc

        if len(payload.embeddings) != len(batch):
            self._logger.warning(
                "ollama_embed_count_mismatch",
                request_id=request.request_id,
                model=request.model,
                expected=len(batch),
                actual=len(payload.embeddings),
            )
            raise EmbeddingError("Embedding provider returned a wrong number of vectors")

        return tuple(tuple(vector) for vector in payload.embeddings)
