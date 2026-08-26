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


def test_cut_does_not_leave_orphaned_tool_result() -> None:
    # Пара «вызов инструмента → результат»: результат без предшествующего
    # вызова невалиден для провайдера, срез смещается дальше.
    history = (
        msg(MessageRole.USER, "запрос"),
        msg(MessageRole.ASSISTANT, "вызов инструмента"),
        msg(MessageRole.TOOL, "результат инструмента"),
        msg(MessageRole.ASSISTANT, "финальный ответ"),
    )

    window = trim_to_window(history, 2)

    assert window == (msg(MessageRole.ASSISTANT, "финальный ответ"),)


def test_window_of_only_tool_results_is_empty() -> None:
    history = (
        msg(MessageRole.TOOL, "результат 1"),
        msg(MessageRole.TOOL, "результат 2"),
    )

    assert trim_to_window(history, 5) == ()


def test_limit_must_be_positive() -> None:
    history = (msg(MessageRole.USER, "a"),)

    with pytest.raises(ValueError, match="history window limit"):
        trim_to_window(history, 0)
