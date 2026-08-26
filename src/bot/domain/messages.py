"""Typed chat messages shared across layers."""

from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    """Roles supported by the chat message representation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    # Результат инструмента; вводится агентным циклом. Окно истории уже
    # учитывает эту роль, чтобы пара «вызов → результат» не разрывалась.
    TOOL = "tool"


@dataclass(frozen=True)
class InferenceMessage:
    """A single message sent to or produced by an inference provider."""

    role: MessageRole
    content: str
