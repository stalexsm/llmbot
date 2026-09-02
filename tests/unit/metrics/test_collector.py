"""Unit tests for the run metrics collector (llm_call + run aggregation)."""

import json
from pathlib import Path

import pytest
import structlog.stdlib

from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceUsage
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.recorder import MetricsRecorder

REQUEST_ID = RequestId("req-1")
MODEL = ModelId("qwen3:1.7b")
SYSTEM = InferenceMessage(role=MessageRole.SYSTEM, content="sys")
USER = InferenceMessage(role=MessageRole.USER, content="user question")
TOOL = InferenceMessage(
    role=MessageRole.TOOL,
    content="result",
    tool_name=ToolId("execute_command"),
)


def make_collector(
    tmp_path: Path, *, input_price: float = 0.0, output_price: float = 0.0
) -> tuple[RunMetricsCollector, Path]:
    directory = tmp_path / "metrics"
    return RunMetricsCollector(
        recorder=MetricsRecorder(directory=directory, logger=structlog.stdlib.get_logger()),
        input_price_per_mtok=input_price,
        output_price_per_mtok=output_price,
    ), directory


def read_lines(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_llm_call_carries_all_issue_fields(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path, input_price=2.0, output_price=8.0)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=350,
        usage=InferenceUsage(prompt_tokens=1000, completion_tokens=500),
        messages=(SYSTEM, USER),
        reached_model=True,
    )

    (line,) = read_lines(directory)
    assert line["kind"] == "llm_call"
    assert line["request_id"] == "req-1"
    assert line["model"] == "qwen3:1.7b"
    assert line["step"] == 1
    assert line["prompt_tokens"] == 1000
    assert line["completion_tokens"] == 500
    assert line["latency_ms"] == 350
    assert line["estimated_cost"] == 1000 / 1_000_000 * 2.0 + 500 / 1_000_000 * 8.0
    assert "timestamp" in line


def test_steps_are_numbered_per_request(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    for _ in range(3):
        collector.record_llm_call(
            request_id=REQUEST_ID,
            model=MODEL,
            latency_ms=10,
            usage=None,
            messages=(SYSTEM,),
            reached_model=True,
        )
    collector.record_llm_call(
        request_id=RequestId("req-2"),
        model=MODEL,
        latency_ms=10,
        usage=None,
        messages=(SYSTEM,),
        reached_model=True,
    )

    assert [line["step"] for line in read_lines(directory)] == [1, 2, 3, 1]


def test_missing_tokens_produce_null_tokens_and_cost(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path, input_price=2.0)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=None,
        messages=(SYSTEM,),
        reached_model=True,
    )

    (line,) = read_lines(directory)
    assert line["prompt_tokens"] is None
    assert line["completion_tokens"] is None
    assert line["estimated_cost"] is None
    assert line["repeated_context_tokens"] is None


def test_run_record_sums_converge_with_llm_call_records(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path, input_price=2.0, output_price=8.0)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=100,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=10),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=200,
        usage=InferenceUsage(prompt_tokens=300, completion_tokens=40),
        messages=(SYSTEM, USER, TOOL),
        reached_model=True,
    )
    collector.finish_run(REQUEST_ID, success=True)

    lines = read_lines(directory)
    calls = [line for line in lines if line["kind"] == "llm_call"]
    (run,) = [line for line in lines if line["kind"] == "run"]
    assert run["steps"] == len(calls) == 2
    assert run["prompt_tokens"] == sum(call["prompt_tokens"] for call in calls)
    assert run["completion_tokens"] == sum(call["completion_tokens"] for call in calls)
    assert run["estimated_cost"] == pytest.approx(sum(call["estimated_cost"] for call in calls))
    assert run["repeated_context_tokens"] == sum(call["repeated_context_tokens"] for call in calls)
    assert run["success"] is True
    assert run["duration_ms"] >= 0
    assert run["model"] == "qwen3:1.7b"
    assert run["request_id"] == "req-1"


def test_run_with_null_tokens_keeps_null_totals(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=None,
        messages=(SYSTEM,),
        reached_model=True,
    )
    collector.finish_run(REQUEST_ID, success=False)

    (run,) = [line for line in read_lines(directory) if line["kind"] == "run"]
    assert run["prompt_tokens"] is None
    assert run["completion_tokens"] is None
    assert run["estimated_cost"] is None
    assert run["repeated_context_tokens"] is None
    assert run["repeated_context_ratio"] is None
    assert run["success"] is False


def test_finish_run_without_llm_calls_writes_empty_run(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.finish_run(REQUEST_ID, success=False)

    (run,) = read_lines(directory)
    assert run["kind"] == "run"
    assert run["steps"] == 0
    assert run["success"] is False


def test_state_is_dropped_after_finish(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=None,
        messages=(SYSTEM,),
        reached_model=True,
    )
    collector.finish_run(REQUEST_ID, success=True)
    collector.finish_run(REQUEST_ID, success=True)

    runs = [line for line in read_lines(directory) if line["kind"] == "run"]
    assert [run["steps"] for run in runs] == [1, 0]


def test_tool_call_writes_estimate_from_output_size(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.record_tool_call(
        request_id=REQUEST_ID,
        tool_name="exec:ls",
        input_size=30,
        output_size=800,
        duration_ms=12,
        succeeded=True,
    )

    (line,) = read_lines(directory)
    assert line["kind"] == "tool_call"
    assert line["request_id"] == "req-1"
    assert line["tool_name"] == "exec:ls"
    assert line["input_size"] == 30
    assert line["output_size"] == 800
    assert line["output_tokens"] == 200
    assert line["duration_ms"] == 12
    assert line["succeeded"] is True
    assert "timestamp" in line


def test_repeated_tokens_calibrated_by_exact_prompt_tokens(tmp_path: Path) -> None:
    """Общий префикс даёт повторные токены > 0, откалиброванные по prompt_tokens."""
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    # Тот же запуск: промпт вырос на TOOL («result», 6 символов); общий префикс
    # — прежние SYSTEM+USER (16 из 22 символов) → 220 * 16 / 22 = 160.
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=220, completion_tokens=1),
        messages=(SYSTEM, USER, TOOL),
        reached_model=True,
    )

    lines = read_lines(directory)
    assert [line["repeated_context_tokens"] for line in lines] == [0, 160]


def test_disjoint_requests_repeat_zero_tokens(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)
    other_system = InferenceMessage(role=MessageRole.SYSTEM, content="other")

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=50, completion_tokens=1),
        messages=(other_system,),
        reached_model=True,
    )

    lines = read_lines(directory)
    assert [line["repeated_context_tokens"] for line in lines] == [0, 0]


def test_run_record_carries_repeated_total_and_ratio(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=220, completion_tokens=1),
        messages=(SYSTEM, USER, TOOL),
        reached_model=True,
    )
    collector.finish_run(REQUEST_ID, success=True)

    (run,) = [line for line in read_lines(directory) if line["kind"] == "run"]
    assert run["repeated_context_tokens"] == 160
    assert run["repeated_context_ratio"] == pytest.approx(160 / 320)


def test_repeat_state_does_not_leak_across_runs(tmp_path: Path) -> None:
    """После finish_run тот же request_id стартует с чистого листа."""
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    collector.finish_run(REQUEST_ID, success=True)
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=300, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    assert [call["repeated_context_tokens"] for call in calls] == [0, 0]


def test_unreached_prompt_is_not_previous_request(tmp_path: Path) -> None:
    """Промпт сбойного вызова модель не видела — предыдущим запросом не становится."""
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=None,
        messages=(SYSTEM, USER),
        reached_model=False,
    )
    # Тот же промпт тем же запуском: сравнение только с последним дошедшим
    # до модели запросом (его ещё нет) — ноль, а не полный повтор.
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=160, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    # У сбойного вызова нет точного счётчика — None; следующий вызов видит 0.
    assert [call["repeated_context_tokens"] for call in calls] == [None, 0]


def test_unreached_call_keeps_last_seen_request(tmp_path: Path) -> None:
    collector, directory = make_collector(tmp_path)

    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=100, completion_tokens=1),
        messages=(SYSTEM, USER),
        reached_model=True,
    )
    # Сбой после успешного шага: виденный префикс — всё ещё первый запрос.
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=None,
        messages=(SYSTEM, USER, TOOL),
        reached_model=False,
    )
    collector.record_llm_call(
        request_id=REQUEST_ID,
        model=MODEL,
        latency_ms=5,
        usage=InferenceUsage(prompt_tokens=220, completion_tokens=1),
        messages=(SYSTEM, USER, TOOL),
        reached_model=True,
    )

    calls = [line for line in read_lines(directory) if line["kind"] == "llm_call"]
    # Третий вызов сравнивается с первым (16 из 22 символов), не со вторым.
    assert [call["repeated_context_tokens"] for call in calls] == [0, None, 160]
