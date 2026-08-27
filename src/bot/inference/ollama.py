"""Ollama infrastructure adapter.

Responsibilities:
    - HTTP communication with a local Ollama instance;
    - request serialization (включая нативный протокол инструментов);
    - response parsing and validation;
    - timeout handling;
    - mapping infrastructure errors to application-level errors.

The adapter knows nothing about Telegram.
"""

import contextlib
import json
import re

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from bot.application.errors import (
    InferenceError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from bot.domain.ids import ToolId
from bot.domain.messages import InferenceMessage
from bot.domain.tools import ToolCall, ToolSpec
from bot.inference.models import InferenceRequest, InferenceResponse

# Служебные размышления модели не должны попадать в контент ответа.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class _ToolCallFunctionPayload(BaseModel):
    """Wire format of a tool call inside a chat message (external boundary only)."""

    name: str
    # Новые сборки Ollama возвращают объект, старые — JSON-строку.
    arguments: dict[str, object] | str = {}


class _ToolCallPayload(BaseModel):
    """Wire format of an OpenAI-style tool call (external boundary only)."""

    function: _ToolCallFunctionPayload


class _ChatMessagePayload(BaseModel):
    """Wire format of an Ollama chat message (external boundary only)."""

    role: str
    content: str = ""
    tool_calls: list[_ToolCallPayload] | None = None


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
        think: bool = False,
    ) -> None:
        self._client = client
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._timeout = httpx.Timeout(timeout_seconds)
        self._logger = logger.bind(component="ollama_inference_provider")
        self._think = think

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
            content=self._strip_thinking(payload.message.content),
            tool_calls=tuple(
                _tool_call_to_domain(call.function) for call in payload.message.tool_calls or ()
            ),
        )

    def _serialize_request(self, request: InferenceRequest) -> dict[str, object]:
        body: dict[str, object] = {
            "model": request.model,
            "messages": [self._serialize_message(message) for message in request.messages],
            "stream": False,
            # Размышления управляются конфигурацией (OLLAMA_THINK): содержание
            # режима при необходимости вырезается из ответа (_strip_thinking),
            # а пользователю попадает только чистый результат.
            "think": self._think,
        }
        if request.tools:
            body["tools"] = [self._serialize_tool(spec) for spec in request.tools]
        return body

    @staticmethod
    def _serialize_message(message: InferenceMessage) -> dict[str, object]:
        payload: dict[str, object] = {"role": message.role.value, "content": message.content}
        if message.tool_name is not None:
            payload["tool_name"] = message.tool_name
        if message.tool_calls:
            payload["tool_calls"] = [
                {"function": {"name": call.name, "arguments": _arguments_to_wire(call.arguments)}}
                for call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _serialize_tool(spec: ToolSpec) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter.name: {
                            "type": parameter.type,
                            "description": parameter.description,
                        }
                        for parameter in spec.parameters
                    },
                    "required": list(spec.required),
                },
            },
        }

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Убрать служебные размышления модели из контента ответа."""
        stripped = _THINK_BLOCK_RE.sub("", content)
        unclosed = stripped.find("<think>")
        if unclosed != -1:
            stripped = stripped[:unclosed]
        return stripped.strip()


def _tool_call_to_domain(function: _ToolCallFunctionPayload) -> ToolCall:
    """Привести вызов инструмента из wire-формата к доменной модели."""
    if isinstance(function.arguments, str):
        encoded = function.arguments
    else:
        encoded = json.dumps(function.arguments, ensure_ascii=False)
    return ToolCall(name=ToolId(function.name), arguments=encoded)


def _arguments_to_wire(arguments: str) -> dict[str, object]:
    """Домен хранит аргументы JSON-строкой; Ollama ждёт объект."""
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
    return {}
