"""Decorator over the InferenceProvider seam: llm_call per model call + run totals."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
import structlog.stdlib

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.application.errors import InferenceUnavailableError
from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
from bot.inference.models import InferenceRequest, InferenceResponse, InferenceUsage
from bot.inference.provider import InferenceProvider
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.provider import MeteredInferenceProvider
from bot.metrics.recorder import MetricsRecorder
from tests.fakes import (
    FailingInferenceProvider,
    MockInferenceProvider,
    ScriptedInferenceProvider,
)

REQUEST_ID = RequestId("req-1")
MODEL = ModelId("qwen3:1.7b")
USER = InferenceMessage(role=MessageRole.USER, content="вопрос")


def make_metering(
    tmp_path: Path, inner: InferenceProvider
) -> tuple[MeteredInferenceProvider, RunMetricsCollector, Path]:
    directory = tmp_path / "metrics"
    collector = RunMetricsCollector(
        recorder=MetricsRecorder(directory=directory, logger=structlog.stdlib.get_logger()),
        input_price_per_mtok=2.0,
        output_price_per_mtok=8.0,
    )
    return MeteredInferenceProvider(inner=inner, collector=collector), collector, directory


def read_lines(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def test_writes_llm_call_for_every_model_call(tmp_path: Path) -> None:
    inner = MockInferenceProvider(response_content="Ответ модели")
    metered, _, directory = make_metering(tmp_path, inner)

    response = await metered.generate(
        InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(USER,))
    )

    assert response.content == "Ответ модели"
    (line,) = read_lines(directory)
    assert line["kind"] == "llm_call"
    assert line["request_id"] == "req-1"
    assert line["model"] == "qwen3:1.7b"
    assert line["latency_ms"] >= 0


async def test_passes_response_usage_to_collector(tmp_path: Path) -> None:
    inner = MockInferenceProvider(
        response_content="Ответ модели",
        usage=InferenceUsage(prompt_tokens=120, completion_tokens=30),
    )
    metered, _, directory = make_metering(tmp_path, inner)

    await metered.generate(InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(USER,)))

    (line,) = read_lines(directory)
    assert line["prompt_tokens"] == 120
    assert line["completion_tokens"] == 30


async def test_failed_call_also_writes_llm_call_and_reraises(tmp_path: Path) -> None:
    inner = FailingInferenceProvider(InferenceUnavailableError("down"))
    metered, _, directory = make_metering(tmp_path, inner)

    with pytest.raises(InferenceUnavailableError):
        await metered.generate(
            InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(USER,))
        )

    (line,) = read_lines(directory)
    assert line["kind"] == "llm_call"
    assert line["prompt_tokens"] is None


async def test_run_totals_converge_with_llm_calls_on_scripted_provider(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Чеклист задания: суммы run сходятся с llm_call того же запуска."""
    inner = ScriptedInferenceProvider(
        [
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="",
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
            ),
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="Готово",
                usage=InferenceUsage(prompt_tokens=250, completion_tokens=25),
            ),
        ]
    )
    metered, collector, directory = make_metering(tmp_path, inner)
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    loop = AgentLoop(
        inference=metered,
        model=MODEL,
        system_prompt="Ты тестовый агент.",
        tools=(exec_tool,),
        step_limit=10,
        logger=logger,
    )

    run_request_id = RequestId(str(uuid4()))
    run = await loop.run(run_request_id, (), USER)

    assert run.final_answer == "Готово"
    collector.finish_run(run_request_id, success=True)

    lines = read_lines(directory)
    calls = [line for line in lines if line["kind"] == "llm_call"]
    (run_record,) = [line for line in lines if line["kind"] == "run"]
    assert [call["step"] for call in calls] == [1, 2]
    assert all(call["request_id"] == str(run_request_id) for call in calls)
    assert run_record["steps"] == len(calls) == 2
    assert run_record["prompt_tokens"] == sum(call["prompt_tokens"] for call in calls) == 350
    assert run_record["completion_tokens"] == sum(call["completion_tokens"] for call in calls) == 35
    assert run_record["success"] is True


async def test_repeated_context_tokens_on_scripted_provider(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Чеклист задания: общий префикс даёт повторные токены > 0, без него — 0."""
    inner = ScriptedInferenceProvider(
        [
            # Шаг 1: первый запрос запуска — повторять нечего.
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="",
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
                tool_calls=(
                    ToolCall(
                        name=ToolId("execute_command"),
                        arguments=json.dumps({"command": "echo hi"}),
                    ),
                ),
            ),
            # Шаг 2: к запросу добавились assistant и результат инструмента —
            # префикс с шагом 1 общий.
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="Готово",
                usage=InferenceUsage(prompt_tokens=250, completion_tokens=10),
            ),
        ]
    )
    metered, collector, directory = make_metering(tmp_path, inner)
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    loop = AgentLoop(
        inference=metered,
        model=MODEL,
        system_prompt="Ты тестовый агент.",
        tools=(exec_tool,),
        step_limit=10,
        logger=logger,
    )

    run_request_id = RequestId(str(uuid4()))
    await loop.run(run_request_id, (), USER)
    collector.finish_run(run_request_id, success=True)

    lines = read_lines(directory)
    calls = [line for line in lines if line["kind"] == "llm_call"]
    (run_record,) = [line for line in lines if line["kind"] == "run"]
    assert calls[0]["repeated_context_tokens"] == 0
    assert 0 < calls[1]["repeated_context_tokens"] < calls[1]["prompt_tokens"]
    assert run_record["repeated_context_tokens"] == sum(
        call["repeated_context_tokens"] for call in calls
    )
    assert run_record["repeated_context_ratio"] == round(
        run_record["repeated_context_tokens"] / run_record["prompt_tokens"], 4
    )


async def test_new_run_starts_with_zero_repeated_context(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Повторный контекст считается только внутри одного Запуска агента."""
    inner = ScriptedInferenceProvider(
        [
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="Готово",
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
            ),
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="Готово",
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
            ),
        ]
    )
    metered, collector, directory = make_metering(tmp_path, inner)
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    loop = AgentLoop(
        inference=metered,
        model=MODEL,
        system_prompt="Ты тестовый агент.",
        tools=(exec_tool,),
        step_limit=10,
        logger=logger,
    )

    first_run = RequestId(str(uuid4()))
    await loop.run(first_run, (), USER)
    collector.finish_run(first_run, success=True)
    second_run = RequestId(str(uuid4()))
    await loop.run(second_run, (), USER)
    collector.finish_run(second_run, success=True)

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    assert [call["repeated_context_tokens"] for call in calls] == [0, 0]


class _FlakyThenOkProvider:
    """Провайдер, роняющий первый вызов и отвечающий на второй."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        if self.calls == 1:
            raise InferenceUnavailableError("down")
        return InferenceResponse(
            request_id=request.request_id,
            content="Готово",
            usage=InferenceUsage(prompt_tokens=160, completion_tokens=1),
        )


async def test_failed_call_prompt_is_not_counted_as_seen(tmp_path: Path) -> None:
    """Промпт упавшего вызова не становится «уже виденным» для следующего."""
    metered, _, directory = make_metering(tmp_path, _FlakyThenOkProvider())
    request = InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(USER,))

    with pytest.raises(InferenceUnavailableError):
        await metered.generate(request)
    await metered.generate(request)

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    # Сбойный вызов без счётчика — None; повторный промпт не засчитан (0).
    assert [call["repeated_context_tokens"] for call in calls] == [None, 0]


async def test_disjoint_requests_on_scripted_provider_repeat_zero(tmp_path: Path) -> None:
    """Чеклист задания: два последовательных запроса без общего префикса — 0."""
    inner = ScriptedInferenceProvider(
        [
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="первый",
                usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
            ),
            InferenceResponse(
                request_id=RequestId("scripted"),
                content="второй",
                usage=InferenceUsage(prompt_tokens=50, completion_tokens=1),
            ),
        ]
    )
    metered, _, directory = make_metering(tmp_path, inner)

    await metered.generate(InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(USER,)))
    other_user = InferenceMessage(role=MessageRole.USER, content="другой вопрос")
    await metered.generate(
        InferenceRequest(request_id=REQUEST_ID, model=MODEL, messages=(other_user,))
    )

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    assert [call["repeated_context_tokens"] for call in calls] == [0, 0]
