"""Typed inference request/response models."""

from dataclasses import dataclass

from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage
from bot.domain.tools import ToolCall, ToolSpec


@dataclass(frozen=True)
class InferenceRequest:
    """A single non-streaming inference request."""

    request_id: RequestId
    model: ModelId
    messages: tuple[InferenceMessage, ...]
    # Описания инструментов нативного протокола: модель может отвечать tool-calls.
    tools: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True)
class InferenceUsage:
    """Счётчики токенов и длительности одного вызова модели.

    Заполняется из ответа провайдера, если тот их сообщает; отсутствующие
    значения остаются ``None``. Единица длительности — наносекунды, как в
    ответе Ollama ``/api/chat``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None


@dataclass(frozen=True)
class InferenceResponse:
    """A completed inference result."""

    request_id: RequestId
    content: str
    # Вызовы инструментов, запрошенные моделью; пусто — финальный ответ.
    tool_calls: tuple[ToolCall, ...] = ()
    # Учёт токенов и времени, если провайдер их сообщил.
    usage: InferenceUsage | None = None
