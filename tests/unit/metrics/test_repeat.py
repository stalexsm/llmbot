"""Unit tests for the repeated-prompt-prefix seam (issue 04)."""

from bot.domain.ids import ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
from bot.metrics.repeat import prompt_chars, repeated_prefix_chars


def msg(role: MessageRole, content: str) -> InferenceMessage:
    return InferenceMessage(role=role, content=content)


def tool_call(arguments: str) -> ToolCall:
    return ToolCall(name=ToolId("execute_command"), arguments=arguments)


def test_prompt_chars_sums_content_and_tool_call_arguments() -> None:
    messages = (
        InferenceMessage(role=MessageRole.ASSISTANT, content="", tool_calls=(tool_call("{}"),)),
        msg(MessageRole.TOOL, "файл_1\nфайл_2"),
    )

    assert prompt_chars(messages) == len("{}") + len("файл_1\nфайл_2")


def test_repeated_prefix_counts_common_leading_messages() -> None:
    shared = (msg(MessageRole.SYSTEM, "система"), msg(MessageRole.USER, "вопрос"))
    current = (*shared, msg(MessageRole.ASSISTANT, "ответ"))

    assert repeated_prefix_chars(shared, current) == len("система") + len("вопрос")


def test_repeated_prefix_stops_at_first_different_message() -> None:
    previous = (msg(MessageRole.SYSTEM, "система"), msg(MessageRole.USER, "старый"))
    current = (msg(MessageRole.SYSTEM, "система"), msg(MessageRole.USER, "новый"))

    assert repeated_prefix_chars(previous, current) == len("система")


def test_disjoint_messages_repeat_nothing() -> None:
    previous = (msg(MessageRole.SYSTEM, "система"), msg(MessageRole.USER, "вопрос"))
    current = (msg(MessageRole.SYSTEM, "другая система"),)

    assert repeated_prefix_chars(previous, current) == 0


def test_first_call_has_no_previous_request() -> None:
    current = (msg(MessageRole.SYSTEM, "система"), msg(MessageRole.USER, "вопрос"))

    assert repeated_prefix_chars((), current) == 0


def test_message_equality_requires_same_tool_calls() -> None:
    with_call = InferenceMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=(tool_call("{}"),),
    )
    without_call = InferenceMessage(role=MessageRole.ASSISTANT, content="")

    assert repeated_prefix_chars((with_call,), (without_call,)) == 0
