"""Test doubles and factories shared across unit and integration tests."""

import json
from uuid import uuid4

from aiogram.types import Message

from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage
from bot.domain.tools import ToolCall
from bot.inference.embeddings import EmbeddingRequest, EmbeddingResponse
from bot.inference.models import InferenceRequest, InferenceResponse, InferenceUsage
from bot.rag.models import EMBEDDING_DIMENSION


class RecordingProgress:
    """Прогресс, записывающий события шагов агента для проверок."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, bool | None]] = []

    async def command_started(self, command: str) -> None:
        self.events.append(("started", command, None))

    async def command_finished(self, command: str, succeeded: bool) -> None:
        self.events.append(("finished", command, succeeded))


class MockInferenceProvider:
    """Deterministic in-memory inference provider for tests.

    Structurally implements ``bot.inference.provider.InferenceProvider``.
    """

    def __init__(
        self,
        response_content: str = "Ответ модели",
        usage: InferenceUsage | None = None,
    ) -> None:
        self.response_content = response_content
        self.usage = usage
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(
            request_id=request.request_id,
            content=self.response_content,
            usage=self.usage,
        )


class ScriptedInferenceProvider:
    """Провайдер с заранее заготовленными ответами по порядку (для агентного цикла)."""

    def __init__(self, responses: list[InferenceResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("scripted inference responses are exhausted")
        return self._responses.pop(0)


def tool_call_arguments(command: str) -> str:
    """JSON-аргументы вызова exec-инструмента для скриптованных ответов."""
    return json.dumps({"command": command}, ensure_ascii=False)


def exec_call_response(*commands: str) -> InferenceResponse:
    """Ответ модели, запрашивающий инструмент exec с указанными командами."""
    return InferenceResponse(
        request_id=RequestId("scripted"),
        content="",
        tool_calls=tuple(
            ToolCall(
                name=ToolId("execute_command"),
                arguments=json.dumps({"command": command}, ensure_ascii=False),
            )
            for command in commands
        ),
    )


def tool_call_response(name: str, arguments: str) -> InferenceResponse:
    """Ответ модели с произвольным вызовом инструмента."""
    return InferenceResponse(
        request_id=RequestId("scripted"),
        content="",
        tool_calls=(ToolCall(name=ToolId(name), arguments=arguments),),
    )


def final_response(text: str) -> InferenceResponse:
    """Финальный ответ модели без вызовов инструментов."""
    return InferenceResponse(request_id=RequestId(str(uuid4())), content=text)


class FailingInferenceProvider:
    """Inference provider that always raises the configured exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise self.error


class SpyMetricsCollector:
    """Фальшивка RunMetrics: запоминает закрытые запуски для проверок."""

    def __init__(self) -> None:
        self.finished: list[tuple[RequestId, bool]] = []

    def record_llm_call(
        self,
        *,
        request_id: RequestId,
        model: ModelId,
        latency_ms: int,
        usage: InferenceUsage | None,
        messages: tuple[InferenceMessage, ...],
        reached_model: bool,
    ) -> None:
        pass

    def finish_run(self, request_id: RequestId, *, success: bool) -> None:
        self.finished.append((request_id, success))


def make_telegram_message(text: str, *, message_id: int = 42, chat_id: int = 100) -> Message:
    """Build a realistic private-chat aiogram Message without any I/O."""
    return Message.model_validate(
        {
            "message_id": message_id,
            "date": 1735689600,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 7, "is_bot": False, "first_name": "Tester"},
            "text": text,
        }
    )


class MockEmbeddingProvider:
    """Детерминированный in-memory провайдер эмбеддингов для тестов.

    Структурно реализует ``bot.inference.embeddings.EmbeddingProvider``:
    вектор текста — устойчивая функция его байтов (одинаковый текст даёт
    одинаковый вектор, похожие — близкие по косинусу).
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.requests: list[EmbeddingRequest] = []
        self._dimension = dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(
            request_id=request.request_id,
            embeddings=tuple(self._vector(text) for text in request.texts),
        )

    def _vector(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimension
        for position, byte in enumerate(text.encode("utf-8")[: self._dimension]):
            values[position] = (byte % 32) / 31.0
        return tuple(values)
