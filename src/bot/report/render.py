"""Рендер отчёта: чистые функции от агрегата и timeline, текст — в stdout.

Формат следует примеру dashboard из задания: блоки «всего токенов», стоимость,
средний запуск, доля повторного контекста, топ классов команд и пошаговый
timeline запуска. Нет данных — прочерк; ноль и неизвестное различаются.
"""

from bot.report.aggregate import Dashboard
from bot.report.timeline import RunTimeline, StepTool, TimelineStep

_WIDTH = 40
_DASHES = "─" * _WIDTH
_LABEL_WIDTH = 28
_VALUE_WIDTH = _WIDTH - _LABEL_WIDTH
_UNKNOWN = "—"


def render_dashboard(dashboard: Dashboard) -> str:
    """Свести агрегат в текст dashboard."""
    lines = [
        "AI AGENT",
        _DASHES,
        "",
        _row("Runs completed", str(dashboard.runs_completed)),
        "",
        "Total tokens",
        _row("  Input", _format_tokens(dashboard.input_tokens)),
        _row("  Output", _format_tokens(dashboard.output_tokens)),
        _row("  Repeated context", _format_tokens(dashboard.repeated_context_tokens)),
        "",
        _row("Estimated cost", _format_cost(dashboard.estimated_cost)),
        "",
        "Average run",
        _row("  Tokens", _format_average(dashboard.avg_run_tokens)),
        _row("  Steps", _format_average(dashboard.avg_run_steps)),
        _row("  Tool calls", _format_average(dashboard.avg_run_tool_calls)),
        "",
        _row("Cache hit rate", _format_percent(dashboard.cache_hit_rate)),
        "",
        "Most expensive tools:",
    ]
    lines.extend(
        _row(f"  {tool.name}", _format_percent(tool.share)) for tool in dashboard.top_tools
    )
    return "\n".join(lines) + "\n"


def render_timeline(timeline: RunTimeline) -> str:
    """Свести timeline запуска в текст."""
    lines = [f"Run {timeline.request_id}", _DASHES, ""]
    for step in timeline.steps:
        lines.append(_step_line(step))
        lines.extend(_tool_line(tool) for tool in step.tools)
        lines.append("")
    lines.append(_DASHES)
    lines.append(_total_line(timeline))
    return "\n".join(lines) + "\n"


def _step_line(step: TimelineStep) -> str:
    tokens = _format_full(step.prompt_tokens)
    return f"Step {step.step:<4} LLM {tokens:>9} tokens"


def _tool_line(tool: StepTool) -> str:
    tokens = _format_full(tool.output_tokens)
    suffix = "" if tool.succeeded else " (failed)"
    return f"{'':<9}{tool.name:<12}{tokens:>9}{suffix}"


def _total_line(timeline: RunTimeline) -> str:
    parts = [
        f"input {_format_tokens(timeline.total_prompt_tokens)}",
        f"output {_format_tokens(timeline.total_completion_tokens)}",
        f"repeated {_format_tokens(timeline.total_repeated_context_tokens)}",
        f"cost {_format_cost(timeline.estimated_cost)}",
        f"duration {_format_duration(timeline.duration_ms)}",
        _format_status(timeline.success),
    ]
    return "Total: " + " · ".join(parts)


def _row(label: str, value: str) -> str:
    return f"{label:<{_LABEL_WIDTH}}{value:>{_VALUE_WIDTH}}"


def _format_tokens(value: int | None) -> str:
    """Компактный формат токенов: 4.2M, 39.4k, 8,200."""
    if value is None:
        return _UNKNOWN
    if value >= 100_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def _format_full(value: int | None) -> str:
    """Полный формат токенов с разделителем тысяч: 8,200."""
    return _UNKNOWN if value is None else f"{value:,}"


def _format_average(value: float | None) -> str:
    if value is None:
        return _UNKNOWN
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.1f}"


def _format_percent(rate: float | None) -> str:
    return _UNKNOWN if rate is None else f"{rate * 100:.0f}%"


def _format_cost(value: float | None) -> str:
    if value is None:
        return _UNKNOWN
    if 0 < abs(value) < 1:
        return f"${value:.6f}"
    return f"${value:,.2f}"


def _format_duration(value_ms: int | None) -> str:
    if value_ms is None:
        return _UNKNOWN
    if value_ms >= 1_000:
        return f"{value_ms / 1_000:.1f}s"
    return f"{value_ms}ms"


def _format_status(success: bool | None) -> str:
    if success is None:
        return _UNKNOWN
    return "success" if success else "failed"
