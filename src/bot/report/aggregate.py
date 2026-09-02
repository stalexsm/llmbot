"""Агрегация событий в сводку dashboard: чистая функция от событий.

Суммарные токены и стоимость берутся из ``llm_call`` (каждый вызов модели,
включая незакрытые запуски), средние по запуску — из агрегированных записей
``run``, топ классов команд — из ``tool_call``. Отсутствие данных — ``None``,
а не ноль: неизвестное количество токенов и ноль — разные вещи.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord
from bot.report.events import MetricEvent
from bot.report.sums import sum_float, sum_int

_TOP_TOOLS_LIMIT = 5


@dataclass(frozen=True)
class ToolUsage:
    """Класс команды в сумме по всем запускам и его доля в токенах инструментов."""

    name: str
    output_tokens: int
    share: float


@dataclass(frozen=True)
class Dashboard:
    """Сводка по всем запускам: блоки dashboard из задания."""

    runs_completed: int
    input_tokens: int | None
    output_tokens: int | None
    repeated_context_tokens: int | None
    estimated_cost: float | None
    avg_run_tokens: float | None
    avg_run_steps: float | None
    avg_run_tool_calls: float | None
    cache_hit_rate: float | None
    top_tools: tuple[ToolUsage, ...]


def build_dashboard(events: Iterable[MetricEvent]) -> Dashboard:
    """Свести события в агрегат; пустое хранилище даёт нулевую сводку."""
    llm_calls: list[LlmCallRecord] = []
    tool_calls: list[ToolCallRecord] = []
    runs: list[RunRecord] = []
    for event in events:
        match event:
            case LlmCallRecord():
                llm_calls.append(event)
            case ToolCallRecord():
                tool_calls.append(event)
            case RunRecord():
                runs.append(event)

    tokens_by_run = _tokens_by_run(llm_calls)
    tools_by_run = _count_tools_by_run(tool_calls, {str(run.request_id) for run in runs})
    return Dashboard(
        runs_completed=len(runs),
        input_tokens=sum_int(call.prompt_tokens for call in llm_calls),
        output_tokens=sum_int(call.completion_tokens for call in llm_calls),
        repeated_context_tokens=sum_int(call.repeated_context_tokens for call in llm_calls),
        estimated_cost=sum_float(call.estimated_cost for call in llm_calls),
        avg_run_tokens=_mean(tokens for tokens in tokens_by_run.values() if tokens is not None),
        avg_run_steps=_mean([float(run.steps) for run in runs]),
        avg_run_tool_calls=(
            _mean([float(tools_by_run.get(str(run.request_id), 0)) for run in runs])
        ),
        cache_hit_rate=_cache_hit_rate(llm_calls),
        top_tools=_top_tools(tool_calls),
    )


def _cache_hit_rate(llm_calls: list[LlmCallRecord]) -> float | None:
    """Доля повторно передаваемого контекста (proxy cache hit rate) по всем вызовам."""
    repeated = sum_int(call.repeated_context_tokens for call in llm_calls)
    prompt = sum_int(call.prompt_tokens for call in llm_calls)
    if repeated is None or not prompt:
        return None
    return repeated / prompt


def _tokens_by_run(llm_calls: list[LlmCallRecord]) -> dict[str, int | None]:
    tokens: dict[str, int | None] = {}
    for call in llm_calls:
        key = str(call.request_id)
        known = [
            value for value in (call.prompt_tokens, call.completion_tokens) if value is not None
        ]
        run_tokens = sum(known) if known else None
        tokens[key] = _add_opt_int(tokens.get(key), run_tokens)
    return tokens


def _count_tools_by_run(
    tool_calls: list[ToolCallRecord], completed_run_ids: set[str]
) -> Counter[str]:
    return Counter(
        str(call.request_id) for call in tool_calls if str(call.request_id) in completed_run_ids
    )


def _top_tools(tool_calls: list[ToolCallRecord]) -> tuple[ToolUsage, ...]:
    totals: Counter[str] = Counter()
    for call in tool_calls:
        totals[call.tool_name] += call.output_tokens
    total = sum(totals.values())
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:_TOP_TOOLS_LIMIT]
    return tuple(
        ToolUsage(name=name, output_tokens=tokens, share=tokens / total if total else 0.0)
        for name, tokens in ordered
    )


def _add_opt_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _mean(values: Iterable[float]) -> float | None:
    known = list(values)
    if not known:
        return None
    return sum(known) / len(known)
