"""Рендер dashboard и timeline: чистые функции, текст — как в задании."""

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord
from bot.report.aggregate import Dashboard, ToolUsage, build_dashboard
from bot.report.render import render_dashboard, render_timeline
from bot.report.timeline import RunTimeline, StepTool, TimelineStep


def test_zero_dashboard_renders_all_blocks_without_crash() -> None:
    dashboard = Dashboard(
        runs_completed=0,
        input_tokens=None,
        output_tokens=None,
        repeated_context_tokens=None,
        estimated_cost=None,
        avg_run_tokens=None,
        avg_run_steps=None,
        avg_run_tool_calls=None,
        cache_hit_rate=None,
        top_tools=(),
    )

    text = render_dashboard(dashboard)

    for block in (
        "AI AGENT",
        "Runs completed",
        "Total tokens",
        "Input",
        "Output",
        "Repeated context",
        "Estimated cost",
        "Average run",
        "Tokens",
        "Steps",
        "Tool calls",
        "Cache hit rate",
        "Most expensive tools:",
    ):
        assert block in text
    assert "—" in text


def test_dashboard_renders_numbers_like_spec_example() -> None:
    dashboard = Dashboard(
        runs_completed=3,
        input_tokens=4_200_000,
        output_tokens=800_000,
        repeated_context_tokens=2_700_000,
        estimated_cost=18.42,
        avg_run_tokens=39_400.0,
        avg_run_steps=8.7,
        avg_run_tool_calls=16.2,
        cache_hit_rate=0.64,
        top_tools=(
            ToolUsage(name="exec:git", output_tokens=400, share=0.41),
            ToolUsage(name="exec:rg", output_tokens=200, share=0.22),
        ),
    )

    text = render_dashboard(dashboard)

    assert "Runs completed" in text
    assert "3" in text
    assert "4.2M" in text
    assert "0.8M" in text
    assert "2.7M" in text
    assert "$18.42" in text
    assert "39.4k" in text
    assert "8.7" in text
    assert "16.2" in text
    assert "64%" in text
    assert "exec:git" in text
    assert "41%" in text
    assert "22%" in text


def test_dashboard_renders_small_cost_with_precision() -> None:
    dashboard = build_dashboard(
        (
            LlmCallRecord(
                timestamp="t",
                request_id=RequestId("r1"),
                model=ModelId("m"),
                step=1,
                prompt_tokens=100,
                completion_tokens=10,
                repeated_context_tokens=None,
                latency_ms=5,
                estimated_cost=0.000123,
            ),
        )
    )

    assert "$0.000123" in render_dashboard(dashboard)


def make_timeline() -> RunTimeline:
    return RunTimeline(
        request_id=RequestId("req-184"),
        model="qwen3:1.7b",
        steps=(
            TimelineStep(
                step=1,
                prompt_tokens=8_200,
                completion_tokens=300,
                repeated_context_tokens=6_000,
                tools=(StepTool(name="exec:rg", output_tokens=3_100, succeeded=True),),
            ),
            TimelineStep(
                step=2,
                prompt_tokens=14_400,
                completion_tokens=400,
                repeated_context_tokens=12_000,
                tools=(StepTool(name="exec:git", output_tokens=1_234, succeeded=False),),
            ),
        ),
        total_prompt_tokens=22_600,
        total_completion_tokens=700,
        total_repeated_context_tokens=18_000,
        estimated_cost=0.0123,
        duration_ms=12_340,
        success=True,
    )


def test_timeline_renders_steps_with_thousands_separators() -> None:
    text = render_timeline(make_timeline())

    assert "Run req-184" in text
    assert "Step 1" in text
    assert "8,200 tokens" in text
    assert "14,400 tokens" in text
    assert "exec:rg" in text
    assert "3,100" in text
    assert "exec:git" in text
    assert "1,234" in text
    assert "failed" in text


def test_timeline_renders_total_line() -> None:
    text = render_timeline(make_timeline())

    assert "22.6k" in text
    assert "18.0k" in text
    assert "$0.012300" in text
    assert "12.3s" in text
    assert "success" in text


def test_timeline_renders_totals_even_without_tool_or_run_records() -> None:
    timeline = RunTimeline(
        request_id=RequestId("r1"),
        model=None,
        steps=(
            TimelineStep(
                step=1,
                prompt_tokens=None,
                completion_tokens=None,
                repeated_context_tokens=None,
                tools=(),
            ),
        ),
        total_prompt_tokens=None,
        total_completion_tokens=None,
        total_repeated_context_tokens=None,
        estimated_cost=None,
        duration_ms=None,
        success=None,
    )

    text = render_timeline(timeline)

    assert "Run r1" in text
    assert "—" in text
