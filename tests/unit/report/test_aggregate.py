"""Агрегация событий в сводку dashboard: чистая функция от событий."""

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord
from bot.report.aggregate import ToolUsage, build_dashboard


def llm(
    request_id: str,
    step: int,
    *,
    prompt: int | None,
    completion: int | None,
    repeated: int | None,
    cost: float | None,
    timestamp: str,
) -> LlmCallRecord:
    return LlmCallRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        model=ModelId("qwen3:1.7b"),
        step=step,
        prompt_tokens=prompt,
        completion_tokens=completion,
        repeated_context_tokens=repeated,
        latency_ms=10,
        estimated_cost=cost,
    )


def run(
    request_id: str,
    *,
    steps: int,
    success: bool,
    cost: float | None,
    timestamp: str,
) -> RunRecord:
    return RunRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        model=ModelId("qwen3:1.7b"),
        steps=steps,
        prompt_tokens=None,
        completion_tokens=None,
        repeated_context_tokens=None,
        repeated_context_ratio=None,
        duration_ms=None,
        success=success,
        estimated_cost=cost,
    )


def tool(
    request_id: str,
    name: str,
    output_tokens: int,
    *,
    timestamp: str,
) -> ToolCallRecord:
    return ToolCallRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        tool_name=name,
        input_size=10,
        output_size=output_tokens * 4,
        output_tokens=output_tokens,
        duration_ms=5,
        succeeded=True,
    )


def test_empty_events_give_zero_dashboard() -> None:
    dashboard = build_dashboard(())

    assert dashboard.runs_completed == 0
    assert dashboard.input_tokens is None
    assert dashboard.output_tokens is None
    assert dashboard.repeated_context_tokens is None
    assert dashboard.estimated_cost is None
    assert dashboard.avg_run_tokens is None
    assert dashboard.avg_run_steps is None
    assert dashboard.avg_run_tool_calls is None
    assert dashboard.cache_hit_rate is None
    assert dashboard.top_tools == ()


def test_totals_come_from_llm_calls_only() -> None:
    events = (
        llm("r1", 1, prompt=1000, completion=100, repeated=600, cost=0.001, timestamp="t1"),
        llm("r1", 2, prompt=2000, completion=200, repeated=1200, cost=0.002, timestamp="t2"),
        # Вызов без токенов от провайдера в суммы не попадает.
        llm("r2", 1, prompt=None, completion=None, repeated=None, cost=None, timestamp="t3"),
    )

    dashboard = build_dashboard(events)

    assert dashboard.input_tokens == 3000
    assert dashboard.output_tokens == 300
    assert dashboard.repeated_context_tokens == 1800
    assert dashboard.estimated_cost == 0.003
    assert dashboard.cache_hit_rate == 0.6


def test_averages_come_from_run_records() -> None:
    events = (
        run("r1", steps=2, success=True, cost=None, timestamp="t1"),
        run("r2", steps=4, success=False, cost=None, timestamp="t2"),
        # Токены для среднего: у r1 известно 3300, у r2 — ничего.
        llm("r1", 1, prompt=1000, completion=100, repeated=0, cost=None, timestamp="t3"),
        llm("r1", 2, prompt=2000, completion=200, repeated=0, cost=None, timestamp="t4"),
        llm("r2", 1, prompt=None, completion=None, repeated=None, cost=None, timestamp="t5"),
    )

    dashboard = build_dashboard(events)

    assert dashboard.runs_completed == 2
    assert dashboard.avg_run_tokens == 3300.0
    assert dashboard.avg_run_steps == 3.0


def test_average_tool_calls_counts_tools_per_completed_run() -> None:
    events = (
        run("r1", steps=1, success=True, cost=None, timestamp="t1"),
        run("r2", steps=1, success=True, cost=None, timestamp="t2"),
        run("r3", steps=1, success=True, cost=None, timestamp="t3"),
        tool("r1", "exec:git", 100, timestamp="t4"),
        tool("r1", "exec:rg", 100, timestamp="t5"),
        tool("r2", "exec:git", 100, timestamp="t6"),
        # Вызовы незавершённого запуска в среднее не попадают.
        tool("r-open", "exec:git", 100, timestamp="t7"),
    )

    dashboard = build_dashboard(events)

    assert dashboard.avg_run_tool_calls == 1.0


def test_top_tools_sorted_by_tokens_with_share() -> None:
    dashboard = build_dashboard(
        (
            tool("r1", "exec:git", 300, timestamp="t1"),
            tool("r1", "exec:rg", 100, timestamp="t2"),
            tool("r2", "exec:git", 100, timestamp="t3"),
        )
    )

    assert dashboard.top_tools == (
        ToolUsage(name="exec:git", output_tokens=400, share=0.8),
        ToolUsage(name="exec:rg", output_tokens=100, share=0.2),
    )


def test_top_tools_without_tokens_have_zero_share() -> None:
    dashboard = build_dashboard((tool("r1", "exec:git", 0, timestamp="t1"),))

    assert dashboard.top_tools == (ToolUsage(name="exec:git", output_tokens=0, share=0.0),)
