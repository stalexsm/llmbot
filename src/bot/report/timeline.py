"""Timeline одного запуска агента: чистая функция от событий.

События запуска (по ``request_id``) сливаются по timestamp: вызов модели
открывает шаг, вызовы инструментов после него относятся к этому шагу.
Вызов инструмента до первого вызова модели (порядок, которого в живых
метриках не бывает) относится к первому шагу, а не теряется.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from bot.domain.ids import RequestId
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord
from bot.report.events import MetricEvent
from bot.report.sums import sum_float, sum_int


@dataclass(frozen=True)
class StepTool:
    """Выполненная команда внутри шага: класс и оценка выходных токенов."""

    name: str
    output_tokens: int
    succeeded: bool


@dataclass(frozen=True)
class TimelineStep:
    """Шаг запуска: токены вызова модели и выполненные команды."""

    step: int
    prompt_tokens: int | None
    completion_tokens: int | None
    repeated_context_tokens: int | None
    tools: tuple[StepTool, ...]


@dataclass(frozen=True)
class RunTimeline:
    """Timeline одного запуска: шаги плюс итог запуска."""

    request_id: RequestId
    model: str | None
    steps: tuple[TimelineStep, ...]
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    total_repeated_context_tokens: int | None
    estimated_cost: float | None
    duration_ms: int | None
    success: bool | None


def build_timeline(events: Iterable[MetricEvent], request_id: RequestId) -> RunTimeline | None:
    """Собрать timeline запуска; неизвестный id даёт ``None``."""
    llm_calls: list[LlmCallRecord] = []
    tool_calls: list[ToolCallRecord] = []
    run_record: RunRecord | None = None
    for event in events:
        if event.request_id != request_id:
            continue
        match event:
            case LlmCallRecord():
                llm_calls.append(event)
            case ToolCallRecord():
                tool_calls.append(event)
            case RunRecord():
                run_record = event
    if not llm_calls and not tool_calls and run_record is None:
        return None

    llm_calls.sort(key=lambda call: (call.timestamp, call.step))
    tool_calls.sort(key=lambda call: call.timestamp)
    return RunTimeline(
        request_id=request_id,
        model=_model(llm_calls, run_record),
        steps=_steps(llm_calls, tool_calls),
        total_prompt_tokens=sum_int(call.prompt_tokens for call in llm_calls),
        total_completion_tokens=sum_int(call.completion_tokens for call in llm_calls),
        total_repeated_context_tokens=sum_int(call.repeated_context_tokens for call in llm_calls),
        estimated_cost=sum_float(call.estimated_cost for call in llm_calls),
        duration_ms=run_record.duration_ms if run_record is not None else None,
        success=run_record.success if run_record is not None else None,
    )


def _steps(
    llm_calls: list[LlmCallRecord], tool_calls: list[ToolCallRecord]
) -> tuple[TimelineStep, ...]:
    steps: list[TimelineStep] = [
        TimelineStep(
            step=call.step,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            repeated_context_tokens=call.repeated_context_tokens,
            tools=(),
        )
        for call in llm_calls
    ]
    for call in tool_calls:
        index = _owning_step(llm_calls, call)
        if index is None:
            continue
        step_tool = StepTool(
            name=call.tool_name, output_tokens=call.output_tokens, succeeded=call.succeeded
        )
        steps[index] = replace(steps[index], tools=(*steps[index].tools, step_tool))
    return tuple(steps)


def _owning_step(llm_calls: list[LlmCallRecord], tool_call: ToolCallRecord) -> int | None:
    """Индекс шага-владельца: последний вызов модели не позже tool_call."""
    for index in range(len(llm_calls) - 1, -1, -1):
        if llm_calls[index].timestamp <= tool_call.timestamp:
            return index
    return 0 if llm_calls else None


def _model(llm_calls: list[LlmCallRecord], run_record: RunRecord | None) -> str | None:
    if run_record is not None and run_record.model is not None:
        return run_record.model
    if llm_calls:
        return llm_calls[-1].model
    return None
