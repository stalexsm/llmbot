"""Unit tests for the chat-session history window."""

import pytest

from bot.domain.messages import InferenceMessage, MessageRole
from bot.sessions.window import trim_to_window


def msg(role: MessageRole, content: str) -> InferenceMessage:
    return InferenceMessage(role=role, content=content)


def test_history_shorter_than_limit_is_returned_unchanged() -> None:
    history = (msg(MessageRole.USER, "a"), msg(MessageRole.ASSISTANT, "b"))

    assert trim_to_window(history, 20) == history


def test_history_longer_than_limit_keeps_last_messages() -> None:
    history = tuple(msg(MessageRole.USER, str(i)) for i in range(25))

    window = trim_to_window(history, 20)

    assert len(window) == 20
    assert window[0].content == "5"
    assert window[-1].content == "24"


def test_limit_must_be_positive() -> None:
    history = (msg(MessageRole.USER, "a"),)

    with pytest.raises(ValueError, match="history window limit"):
        trim_to_window(history, 0)
