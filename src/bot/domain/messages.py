"""Typed chat messages shared across layers."""

from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    """Roles supported by the chat message representation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class InferenceMessage:
    """A single message sent to or produced by an inference provider."""

    role: MessageRole
    content: str
