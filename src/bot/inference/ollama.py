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
from bot.inference.models import InferenceRequest, InferenceResponse, InferenceUsage

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
    # Usage-поля /api/chat: у старых сборок и в незавершённых ответах отсутствуют.
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaInferenceProvider:
    """Inference backed by the Ollama ``/api/chat`` endpoint (non-streaming)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_seconds: float,
        logger: structlog.stdlib.BoundLogger,
        think: bool = False,
        num_ctx: int = 8192,
    ) -> None:
        self._client = client
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._timeout = httpx.Timeout(timeout_seconds)
        self._logger = logger.bind(component="ollama_inference_provider")
        self._think = think
        self._num_ctx = num_ctx

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
            usage=_usage_to_domain(payload),
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
            # Контекст на запрос (OLLAMA_NUM_CTX): перекрывает дефолт сервера
            # OLLAMA_CONTEXT_LENGTH независимо от способа его запуска.
            "options": {"num_ctx": self._num_ctx},
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


def _usage_to_domain(payload: _ChatResponsePayload) -> InferenceUsage | None:
    """Привести usage-поля ответа к доменной модели; без них — ``None``."""
    if all(
        value is None
        for value in (
            payload.total_duration,
            payload.load_duration,
            payload.prompt_eval_count,
            payload.prompt_eval_duration,
            payload.eval_count,
            payload.eval_duration,
        )
    ):
        return None
    return InferenceUsage(
        prompt_tokens=payload.prompt_eval_count,
        completion_tokens=payload.eval_count,
        total_duration_ns=payload.total_duration,
        load_duration_ns=payload.load_duration,
        prompt_eval_duration_ns=payload.prompt_eval_duration,
        eval_duration_ns=payload.eval_duration,
    )


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
