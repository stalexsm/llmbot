"""Чтение JSONL-метрик: годные строки — события, битые — пропускаются."""

import json
from pathlib import Path

import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord
from bot.metrics.recorder import MetricsRecorder
from bot.report.events import read_events


def call_record(step: int = 1) -> LlmCallRecord:
    return LlmCallRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        request_id=RequestId("req-1"),
        model=ModelId("qwen3:1.7b"),
        step=step,
        prompt_tokens=100,
        completion_tokens=20,
        repeated_context_tokens=60,
        latency_ms=350,
        estimated_cost=0.0001,
    )


def tool_record() -> ToolCallRecord:
    return ToolCallRecord(
        timestamp="2026-01-01T00:00:01+00:00",
        request_id=RequestId("req-1"),
        tool_name="exec:git",
        input_size=10,
        output_size=400,
        output_tokens=100,
        duration_ms=25,
        succeeded=True,
    )


def run_record() -> RunRecord:
    return RunRecord(
        timestamp="2026-01-01T00:00:02+00:00",
        request_id=RequestId("req-1"),
        model=ModelId("qwen3:1.7b"),
        steps=2,
        prompt_tokens=200,
        completion_tokens=40,
        repeated_context_tokens=120,
        repeated_context_ratio=0.6,
        duration_ms=900,
        success=True,
        estimated_cost=0.0002,
    )


def test_missing_file_yields_no_events(tmp_path: Path) -> None:
    assert read_events(tmp_path / "absent" / "events.jsonl") == ()


def test_roundtrip_of_records_written_by_recorder(
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "metrics"
    recorder = MetricsRecorder(directory=directory, logger=logger)
    written = [call_record(step=1), tool_record(), run_record()]
    for record in written:
        recorder.record(record)

    assert read_events(directory / "events.jsonl") == tuple(written)


def test_broken_and_unknown_lines_are_skipped(tmp_path: Path) -> None:
    lines = [
        "not json at all",
        json.dumps({"kind": "future_kind", "request_id": "req-9"}),
        json.dumps({"kind": "llm_call", "prompt_tokens": "many"}),
        json.dumps(call_record(step=1).to_payload()),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = read_events(path)

    assert events == (call_record(step=1),)
