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
class InferenceResponse:
    """A completed inference result."""

    request_id: RequestId
    content: str
    # Вызовы инструментов, запрошенные моделью; пусто — финальный ответ.
    tool_calls: tuple[ToolCall, ...] = ()
