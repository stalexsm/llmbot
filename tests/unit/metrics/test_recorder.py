"""Unit tests for the append-only metrics JSONL store."""

import json
from pathlib import Path

import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord, RunRecord
from bot.metrics.recorder import MetricsRecorder


def make_call_record(step: int = 1) -> LlmCallRecord:
    return LlmCallRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        request_id=RequestId("req-1"),
        model=ModelId("qwen3:1.7b"),
        step=step,
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=350,
        estimated_cost=0.0001,
    )


def make_run_record() -> RunRecord:
    return RunRecord(
        timestamp="2026-01-01T00:00:01+00:00",
        request_id=RequestId("req-1"),
        model=ModelId("qwen3:1.7b"),
        steps=2,
        prompt_tokens=200,
        completion_tokens=40,
        duration_ms=900,
        success=True,
        estimated_cost=0.0002,
    )


def read_lines(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_creates_directory_and_appends_record(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    recorder = MetricsRecorder(directory=tmp_path / "metrics", logger=logger)

    recorder.record(make_call_record())

    lines = read_lines(tmp_path / "metrics")
    assert len(lines) == 1
    assert lines[0]["kind"] == "llm_call"
    assert lines[0]["request_id"] == "req-1"


def test_records_survive_restart_and_are_never_rewritten(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    directory = tmp_path / "metrics"
    first = MetricsRecorder(directory=directory, logger=logger)
    first.record(make_call_record(step=1))
    first.record(make_call_record(step=2))

    # «Рестарт»: новый экземпляр хранилища поверх тех же файлов.
    restarted = MetricsRecorder(directory=directory, logger=logger)
    restarted.record(make_run_record())

    lines = read_lines(directory)
    assert [record["kind"] for record in lines] == ["llm_call", "llm_call", "run"]
    assert [record.get("step") for record in lines] == [1, 2, None]


def test_structlog_duplicate_correlates_with_jsonl_by_request_id(
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "metrics"
    recorder = MetricsRecorder(directory=directory, logger=logger)

    recorder.record(make_call_record())
    recorder.record(make_run_record())

    events = {call.kwargs["event"]: call.kwargs for call in capturing_logger.calls}
    assert set(events) == {"metrics_llm_call", "metrics_run_finished"}
    lines = read_lines(directory)
    for line in lines:
        expected_event = (
            "metrics_llm_call" if line["kind"] == "llm_call" else "metrics_run_finished"
        )
        duplicate = events[expected_event]
        assert duplicate["request_id"] == line["request_id"]
        assert duplicate["model"] == line["model"]


def test_no_event_carries_message_content(
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "metrics"
    recorder = MetricsRecorder(directory=directory, logger=logger)

    recorder.record(make_call_record())
    recorder.record(make_run_record())

    allowed_keys = {
        "event",
        "level",
        "component",
        "timestamp",
        "kind",
        "request_id",
        "model",
        "step",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "duration_ms",
        "steps",
        "success",
        "estimated_cost",
    }
    for line in read_lines(directory):
        assert set(line) <= allowed_keys
    for call in capturing_logger.calls:
        assert set(call.kwargs) <= allowed_keys


def test_storage_failure_never_raises(
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
    tmp_path: Path,
) -> None:
    # Путь-файл вместо каталога: mkdir упадёт с OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    recorder = MetricsRecorder(directory=blocker / "metrics", logger=logger)

    recorder.record(make_call_record())

    logged = [call.kwargs.get("event") for call in capturing_logger.calls]
    assert "metrics_storage_failed" in logged
