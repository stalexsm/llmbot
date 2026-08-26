"""Unit tests for the agentic loop on a scripted inference provider."""

from pathlib import Path
from uuid import uuid4

import pytest
import structlog.stdlib

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.application.errors import EmptyInferenceResponseError
from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
from bot.inference.provider import InferenceProvider
from tests.fakes import (
    ScriptedInferenceProvider,
    exec_call_response,
    final_response,
    tool_call_response,
)

USER = InferenceMessage(role=MessageRole.USER, content="покажи файлы")


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, bool | None]] = []

    async def command_started(self, command: str) -> None:
        self.events.append(("started", command, None))

    async def command_finished(self, command: str, succeeded: bool) -> None:
        self.events.append(("finished", command, succeeded))


def make_loop(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    tmp_path: Path,
    *,
    step_limit: int = 10,
) -> AgentLoop:
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    return AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt="Ты тестовый агент.",
        tools=(exec_tool,),
        step_limit=step_limit,
        logger=logger,
    )


async def test_simple_question_is_answered_in_one_step(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("Простой ответ")])
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Простой ответ"
    assert run.steps_used == 1
    assert run.stopped_by_limit is False
    # Обмен для сессии: сообщение пользователя + финальный ответ, без служебных системных.
    assert run.exchange == (
        USER,
        InferenceMessage(role=MessageRole.ASSISTANT, content="Простой ответ"),
    )
    # Единственный запрос описывает инструмент exec и содержит системный промпт.
    request = provider.requests[0]
    assert request.tools[0].name == "exec"
    assert [message.role for message in request.messages] == [MessageRole.SYSTEM, MessageRole.USER]


async def test_tool_call_then_final_answer(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo привет"), final_response("Готово: привет")]
    )
    loop = make_loop(logger, provider, tmp_path)
    progress = RecordingProgress()

    run = await loop.run(RequestId(str(uuid4())), (), USER, progress)

    assert run.final_answer == "Готово: привет"
    assert run.steps_used == 2
    # Второй запрос видит пару «вызов → результат» и системный промпт в начале.
    second_request = provider.requests[1]
    assert [message.role for message in second_request.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assistant_call = second_request.messages[2]
    assert assistant_call.tool_calls == (
        ToolCall(name=ToolId("exec"), arguments='{"command": "echo привет"}'),
    )
    tool_result = second_request.messages[3]
    assert tool_result.tool_name is not None
    assert "привет" in tool_result.content
    assert "exit_code: 0" in tool_result.content
    # Обмен для сессии содержит связанную пару.
    assert run.exchange == (
        USER,
        assistant_call,
        tool_result,
        InferenceMessage(role=MessageRole.ASSISTANT, content="Готово: привет"),
    )
    assert progress.events == [
        ("started", "echo привет", None),
        ("finished", "echo привет", True),
    ]


async def test_failed_command_is_visible_to_model(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo беда >&2; exit 3"), final_response("Исправился")]
    )
    loop = make_loop(logger, provider, tmp_path)

    await loop.run(RequestId(str(uuid4())), (), USER)

    tool_result = provider.requests[1].messages[3]
    assert "exit_code: 3" in tool_result.content
    assert "беда" in tool_result.content


async def test_step_limit_stops_loop_with_complete_pairs(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Провайдер бесконечно просит exec: цикл обязан остановиться по лимиту шагов.
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo раз"), exec_call_response("echo два")]
    )
    loop = make_loop(logger, provider, tmp_path, step_limit=2)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.stopped_by_limit is True
    assert run.final_answer is None
    assert run.steps_used == 2
    assert len(provider.requests) == 2
    # Обмен: user + две полные пары «вызов → результат».
    roles = [message.role for message in run.exchange]
    assert roles == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    # Все tool-сообщения связаны с инструментом exec.
    for message in run.exchange:
        if message.role is MessageRole.TOOL:
            assert message.tool_name is not None


async def test_empty_final_answer_raises(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("   ")])
    loop = make_loop(logger, provider, tmp_path)

    with pytest.raises(EmptyInferenceResponseError):
        await loop.run(RequestId(str(uuid4())), (), USER)


async def test_unknown_tool_returns_error_to_model(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [tool_call_response("нетакого", "{}"), final_response("Понял, больше не буду")]
    )
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Понял, больше не буду"
    tool_result = provider.requests[1].messages[3]
    assert "нетакого" in tool_result.content


async def test_history_precedes_user_message(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("Ответ")])
    loop = make_loop(logger, provider, tmp_path)
    history = (
        InferenceMessage(role=MessageRole.USER, content="старый вопрос"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="старый ответ"),
    )

    await loop.run(RequestId(str(uuid4())), history, USER)

    assert [message.content for message in provider.requests[0].messages] == [
        "Ты тестовый агент.",
        "старый вопрос",
        "старый ответ",
        "покажи файлы",
    ]


async def test_no_progress_events_for_direct_answer(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("Ответ")])
    loop = make_loop(logger, provider, tmp_path)
    progress = RecordingProgress()

    await loop.run(RequestId(str(uuid4())), (), USER, progress)

    assert progress.events == []


async def test_null_progress_is_used_by_default(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([exec_call_response("echo тихо"), final_response("Ответ")])
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)  # progress не передан

    assert run.final_answer == "Ответ"
