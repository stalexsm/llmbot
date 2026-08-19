"""Typed inference request/response models."""

from dataclasses import dataclass

from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage


@dataclass(frozen=True)
class InferenceRequest:
    """A single non-streaming inference request."""

    request_id: RequestId
    model: ModelId
    messages: tuple[InferenceMessage, ...]


@dataclass(frozen=True)
class InferenceResponse:
    """A completed inference result."""

    request_id: RequestId
    content: str
