"""Юнит-тесты headless-раннера бенчмарка на скриптованном провайдере."""

import json
from uuid import uuid4

import structlog
import structlog.stdlib

from bot.agent.loop import AgentLoop
from bot.benchmark.runner import BenchmarkRunner
from bot.benchmark.tasks import BenchmarkTask, Checker, TaskKind, tool_call_arguments
from bot.benchmark.tracker import TokenTrackingProvider
from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.tools import ToolCall
from bot.inference.models import InferenceResponse, InferenceUsage
from bot.inference.provider import InferenceProvider
from tests.fakes import FailingInferenceProvider, ScriptedInferenceProvider, final_response

STEP_LIMIT = 5


def make_task(
    task_id: str = "t01",
    checker: Checker = lambda answer: "готово" in answer.lower(),
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        kind=TaskKind.EXEC,
        prompt="сделай что-нибудь",
        checker=checker,
    )


def make_runner(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
) -> BenchmarkRunner:
    tracker = TokenTrackingProvider(inner=provider)
    loop = AgentLoop(
        inference=tracker,
        model=ModelId("qwen3:1.7b"),
        system_prompt="Ты тестовый агент.",
        tools=(),  # exec-поведение в этих тестах не нужно
        step_limit=STEP_LIMIT,
        logger=logger,
    )
    return BenchmarkRunner(agent=loop, tracker=tracker, logger=logger)


def tool_call_step() -> InferenceResponse:
    return InferenceResponse(
        request_id=RequestId(str(uuid4())),
        content="",
        tool_calls=(ToolCall(name=ToolId("execute_command"), arguments='{"command": "echo hi"}'),),
    )


async def test_successful_task_counts_tokens_and_checker(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    task = make_task()
    provider = ScriptedInferenceProvider(
        [
            InferenceResponse(
                request_id=RequestId(str(uuid4())),
                content="",
                tool_calls=(
                    ToolCall(
                        name=ToolId("execute_command"),
                        arguments='{"command": "echo hi"}',
                    ),
                ),
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
            ),
            InferenceResponse(
                request_id=RequestId(str(uuid4())),
                content="Готово: результат",
                usage=InferenceUsage(prompt_tokens=150, completion_tokens=20),
            ),
        ]
    )
    runner = make_runner(logger, provider)

    result = await runner.run_task(task)

    assert result.success is True
    assert result.answer_correct is True
    assert result.steps_used == 2
    assert result.prompt_tokens == 250
    assert result.completion_tokens == 30
    assert result.error is None


async def test_checker_failure_is_recorded_but_success_holds(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    provider = ScriptedInferenceProvider([final_response("совсем не то")])
    runner = make_runner(logger, provider)

    result = await runner.run_task(make_task())

    assert result.success is True
    assert result.answer_correct is False


async def test_stopped_by_limit_is_not_success(logger: structlog.stdlib.BoundLogger) -> None:
    # Модель вечно требует инструменты, которых нет: цикл упирается в лимит шагов.
    responses = [tool_call_step() for _ in range(STEP_LIMIT + 1)]
    provider = ScriptedInferenceProvider(responses)
    runner = make_runner(logger, provider)

    result = await runner.run_task(make_task())

    assert result.success is False
    assert result.steps_used == STEP_LIMIT


async def test_provider_exception_does_not_break_the_run(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    runner = make_runner(logger, FailingInferenceProvider(RuntimeError("ollama down")))

    result = await runner.run_task(make_task())

    assert result.success is False
    assert result.answer_correct is False
    assert result.error is not None
    assert "RuntimeError" in result.error


async def test_run_all_builds_summary(logger: structlog.stdlib.BoundLogger) -> None:
    ok = InferenceResponse(
        request_id=RequestId(str(uuid4())),
        content="готово",
        usage=InferenceUsage(prompt_tokens=10, completion_tokens=5),
    )
    provider = ScriptedInferenceProvider([ok, ok])
    runner = make_runner(logger, provider)
    tasks = (make_task("t01"), make_task("t02"))

    summary = await runner.run_all(tasks)

    assert summary.total == 2
    assert summary.succeeded == 2
    assert summary.success_rate == 1.0
    assert summary.answered == 2
    assert summary.avg_prompt_tokens == 10.0
    assert summary.avg_completion_tokens == 5.0
    assert summary.avg_total_tokens == 15.0


def test_tool_call_arguments_helper_produces_json_object() -> None:
    arguments = json.loads(tool_call_arguments("echo hi"))
    assert arguments == {"command": "echo hi"}


async def test_request_id_is_derived_from_task_id(logger: structlog.stdlib.BoundLogger) -> None:
    provider = ScriptedInferenceProvider([final_response("готово")])
    runner = make_runner(logger, provider)

    await runner.run_task(make_task("custom-id"))

    assert provider.requests[0].request_id == "bench-custom-id"
