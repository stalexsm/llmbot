"""Ollama infrastructure adapter.

Responsibilities:
    - HTTP communication with a local Ollama instance;
    - request serialization;
    - response parsing and validation;
    - timeout handling;
    - mapping infrastructure errors to application-level errors.

The adapter knows nothing about Telegram.
"""

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from bot.application.errors import (
    InferenceError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from bot.inference.models import InferenceRequest, InferenceResponse


class _ChatMessagePayload(BaseModel):
    """Wire format of an Ollama chat message (external boundary only)."""

    role: str
    content: str


class _ChatResponsePayload(BaseModel):
    """Wire format of an Ollama chat completion response (external boundary only)."""

    message: _ChatMessagePayload


class OllamaInferenceProvider:
    """Inference backed by the Ollama ``/api/chat`` endpoint (non-streaming)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_seconds: float,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._client = client
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._timeout = httpx.Timeout(timeout_seconds)
        self._logger = logger.bind(component="ollama_inference_provider")

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        try:
            http_response = await self._client.post(
                self._chat_url,
                json=self._serialize_request(request),
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            self._logger.warning(
                "ollama_request_timed_out",
                request_id=request.request_id,
                model=request.model,
            )
            raise InferenceTimeoutError("Inference request timed out") from exc
        except httpx.HTTPError as exc:
            self._logger.warning(
                "ollama_request_failed",
                request_id=request.request_id,
                model=request.model,
            )
            raise InferenceUnavailableError("Inference provider is unreachable") from exc

        if http_response.status_code != httpx.codes.OK:
            self._logger.warning(
                "ollama_http_error",
                request_id=request.request_id,
                model=request.model,
                status_code=http_response.status_code,
            )
            raise InferenceUnavailableError("Inference provider returned an error status")

        try:
            payload = _ChatResponsePayload.model_validate_json(http_response.text)
        except ValidationError as exc:
            self._logger.warning(
                "ollama_invalid_response",
                request_id=request.request_id,
                model=request.model,
            )
            raise InferenceError("Inference provider returned a malformed response") from exc

        return InferenceResponse(
            request_id=request.request_id,
            content=payload.message.content,
        )

    def _serialize_request(self, request: InferenceRequest) -> dict[str, object]:
        return {
            "model": request.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "stream": False,
        }
