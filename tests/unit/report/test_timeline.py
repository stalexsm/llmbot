"""Timeline одного запуска: чистая функция от событий."""

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord
from bot.report.timeline import StepTool, build_timeline


def llm(request_id: str, step: int, *, timestamp: str, model: str = "qwen3:1.7b") -> LlmCallRecord:
    return LlmCallRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        model=ModelId(model),
        step=step,
        prompt_tokens=1000 * step,
        completion_tokens=100 * step,
        repeated_context_tokens=600 * step,
        latency_ms=10,
        estimated_cost=0.001 * step,
    )


def tool(request_id: str, name: str, *, timestamp: str, succeeded: bool = True) -> ToolCallRecord:
    return ToolCallRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        tool_name=name,
        input_size=10,
        output_size=400,
        output_tokens=100,
        duration_ms=5,
        succeeded=succeeded,
    )


def run_record(request_id: str, *, timestamp: str) -> RunRecord:
    return RunRecord(
        timestamp=timestamp,
        request_id=RequestId(request_id),
        model=ModelId("qwen3:1.7b"),
        steps=2,
        prompt_tokens=3000,
        completion_tokens=300,
        repeated_context_tokens=1800,
        repeated_context_ratio=0.6,
        duration_ms=4000,
        success=True,
        estimated_cost=0.003,
    )


def test_unknown_request_id_gives_no_timeline() -> None:
    assert build_timeline((run_record("r1", timestamp="t1"),), RequestId("nope")) is None


def test_steps_carry_llm_tokens_and_their_tool_calls() -> None:
    events = (
        llm("r1", 1, timestamp="00:01"),
        tool("r1", "exec:rg", timestamp="00:02"),
        tool("r1", "exec:git", timestamp="00:03"),
        llm("r1", 2, timestamp="00:04"),
        tool("r1", "exec:cat", timestamp="00:05"),
        run_record("r1", timestamp="00:06"),
    )

    timeline = build_timeline(events, RequestId("r1"))

    assert timeline is not None
    assert timeline.request_id == RequestId("r1")
    assert [(step.step, step.tools) for step in timeline.steps] == [
        (
            1,
            (
                StepTool(name="exec:rg", output_tokens=100, succeeded=True),
                StepTool(name="exec:git", output_tokens=100, succeeded=True),
            ),
        ),
        (2, (StepTool(name="exec:cat", output_tokens=100, succeeded=True),)),
    ]
    assert timeline.steps[0].prompt_tokens == 1000
    assert timeline.steps[0].completion_tokens == 100
    assert timeline.steps[0].repeated_context_tokens == 600
    assert timeline.total_prompt_tokens == 3000
    assert timeline.total_completion_tokens == 300
    assert timeline.total_repeated_context_tokens == 1800
    assert timeline.estimated_cost == 0.003
    assert timeline.model == "qwen3:1.7b"
    assert timeline.duration_ms == 4000
    assert timeline.success is True


def test_tool_call_before_first_llm_call_attaches_to_first_step() -> None:
    events = (
        tool("r1", "exec:git", timestamp="00:00"),
        llm("r1", 1, timestamp="00:01"),
    )

    timeline = build_timeline(events, RequestId("r1"))

    assert timeline is not None
    assert [step.step for step in timeline.steps] == [1]
    assert timeline.steps[0].tools[0].name == "exec:git"


def test_events_of_other_runs_are_ignored() -> None:
    events = (
        llm("r1", 1, timestamp="00:01"),
        llm("r2", 1, timestamp="00:02"),
        tool("r2", "exec:rg", timestamp="00:03"),
    )

    timeline = build_timeline(events, RequestId("r1"))

    assert timeline is not None
    assert [step.step for step in timeline.steps] == [1]
    assert timeline.steps[0].tools == ()


def test_open_run_without_run_record_still_renders_totals_from_llm_calls() -> None:
    events = (llm("r1", 1, timestamp="00:01"),)

    timeline = build_timeline(events, RequestId("r1"))

    assert timeline is not None
    assert timeline.success is None
    assert timeline.duration_ms is None
    assert timeline.total_prompt_tokens == 1000
    assert timeline.estimated_cost == 0.001
