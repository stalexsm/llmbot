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
    RecordingProgress,
    ScriptedInferenceProvider,
    exec_call_response,
    final_response,
    tool_call_response,
)

USER = InferenceMessage(role=MessageRole.USER, content="покажи файлы")


def make_loop(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    tmp_path: Path,
    *,
    step_limit: int = 10,
    keep_steps: int = 3,
) -> AgentLoop:
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    return AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt="Ты тестовый агент.",
        tools=(exec_tool,),
        step_limit=step_limit,
        keep_steps=keep_steps,
        logger=logger,
    )


def _tool_messages(request_messages: tuple[InferenceMessage, ...]) -> list[InferenceMessage]:
    return [message for message in request_messages if message.role is MessageRole.TOOL]


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
    # Единственный запрос описывает инструмент execute_command и содержит
    # системный промпт со свежей датой.
    request = provider.requests[0]
    assert request.tools[0].name == "execute_command"
    assert [message.role for message in request.messages] == [MessageRole.SYSTEM, MessageRole.USER]
    assert "Сегодняшняя дата:" in request.messages[0].content


async def test_tool_outputs_older_than_keep_steps_are_compacted(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("echo один"),
            exec_call_response("false"),
            exec_call_response("echo три"),
            exec_call_response("echo четыре"),
            final_response("Готово"),
        ]
    )
    loop = make_loop(logger, provider, tmp_path, step_limit=5, keep_steps=1)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Готово"
    # Шаг 4 (запрос index 3): вывод шага 3 полный, шагов 1-2 — схлопнут в сигнатуру.
    step4_tools = _tool_messages(provider.requests[3].messages)
    assert step4_tools[0].content == "echo один → ok"
    assert step4_tools[1].content == "false → ошибка"
    assert step4_tools[2].content.startswith("exit_code")
    assert "три\n" in step4_tools[2].content
    # Шаг 5: шаг 4 ещё в окне — полный вывод; шаги 1-3 схлопнуты.
    step5_tools = _tool_messages(provider.requests[4].messages)
    assert step5_tools[0].content == "echo один → ok"
    assert step5_tools[1].content == "false → ошибка"
    assert step5_tools[2].content == "echo три → ok"
    assert step5_tools[3].content.startswith("exit_code")
    assert "четыре\n" in step5_tools[3].content
    # Записанный обмен запуска не тронут: полные выводы остаются в exchange.
    exchange_tools = [message for message in run.exchange if message.role is MessageRole.TOOL]
    assert "один\n" in exchange_tools[0].content


async def test_keep_steps_zero_disables_compaction(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("echo один"),
            exec_call_response("echo два"),
            exec_call_response("echo три"),
            exec_call_response("echo четыре"),
            final_response("Готово"),
        ]
    )
    loop = make_loop(logger, provider, tmp_path, step_limit=5, keep_steps=0)

    await loop.run(RequestId(str(uuid4())), (), USER)

    for request in provider.requests:
        for message in _tool_messages(request.messages):
            assert "exit_code" in message.content


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
        ToolCall(name=ToolId("execute_command"), arguments='{"command": "echo привет"}'),
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


async def test_empty_final_answer_retries_then_succeeds(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Первые два ответа пустые — цикл повторяет запрос и добирает содержательный.
    provider = ScriptedInferenceProvider(
        [final_response("   "), final_response(""), final_response("Ответ")]
    )
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Ответ"
    assert len(provider.requests) == 3


async def test_empty_final_answer_raises_after_retries_exhausted(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response(" ")] * 3)
    loop = make_loop(logger, provider, tmp_path)

    with pytest.raises(EmptyInferenceResponseError):
        await loop.run(RequestId(str(uuid4())), (), USER)

    # Исчерпание повторов: три попытки (первая + два повтора), не больше.
    assert len(provider.requests) == 3


async def test_tool_echo_is_stripped_from_final_answer(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Модель переписала сырой результат инструмента перед собственной сводкой:
    # пользователю и в сессию должна уйти только сводка.
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("echo погода"),
            final_response("exit_code: 0\nstdout:\nпогода\n\nСводка: погода получена"),
        ]
    )
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Сводка: погода получена"
    # В сессию тоже пишется очищенный ответ, иначе история учит модель эху.
    assert run.exchange[-1] == InferenceMessage(
        role=MessageRole.ASSISTANT, content="Сводка: погода получена"
    )


async def test_pure_tool_echo_is_retried_as_empty_answer(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Ответ целиком из эха результата — после очистки пустота, шаг повторяется.
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("echo погода"),
            final_response("exit_code: 0\nstdout:\nпогода\n\nstderr:\n"),
            final_response("Сегодня погода"),
        ]
    )
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Сегодня погода"
    assert len(provider.requests) == 3


async def test_answer_built_from_tool_data_is_untouched(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Ответ использует данные из вывода, но не копирует его сырьём — не трогаем.
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo 18"), final_response("Сейчас 18 градусов, ясно")]
    )
    loop = make_loop(logger, provider, tmp_path)

    run = await loop.run(RequestId(str(uuid4())), (), USER)

    assert run.final_answer == "Сейчас 18 градусов, ясно"


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

    system_content = provider.requests[0].messages[0].content
    assert system_content.startswith("Ты тестовый агент.")
    assert "Сегодняшняя дата:" in system_content
    assert [message.content for message in provider.requests[0].messages][1:] == [
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
