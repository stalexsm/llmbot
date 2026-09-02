"""CLI-dashboard: чтение файла метрик, агрегат и timeline по --task."""

import json
from pathlib import Path

import pytest

from bot.report.cli import main


def write_events(path: Path, payloads: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    return path


def llm_payload(request_id: str, step: int, timestamp: str) -> dict:
    return {
        "kind": "llm_call",
        "timestamp": timestamp,
        "request_id": request_id,
        "model": "qwen3:1.7b",
        "step": step,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "repeated_context_tokens": 600,
        "latency_ms": 10,
        "estimated_cost": 0.001,
    }


def tool_payload(request_id: str, name: str, timestamp: str) -> dict:
    return {
        "kind": "tool_call",
        "timestamp": timestamp,
        "request_id": request_id,
        "tool_name": name,
        "input_size": 10,
        "output_size": 400,
        "output_tokens": 100,
        "duration_ms": 5,
        "succeeded": True,
    }


def run_payload(request_id: str, timestamp: str) -> dict:
    return {
        "kind": "run",
        "timestamp": timestamp,
        "request_id": request_id,
        "model": "qwen3:1.7b",
        "steps": 2,
        "prompt_tokens": 2000,
        "completion_tokens": 200,
        "repeated_context_tokens": 1200,
        "repeated_context_ratio": 0.6,
        "duration_ms": 900,
        "success": True,
        "estimated_cost": 0.002,
    }


def test_dashboard_printed_from_events_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events_file = write_events(
        tmp_path / "events.jsonl",
        [
            llm_payload("r1", 1, "00:01"),
            tool_payload("r1", "exec:git", "00:02"),
            run_payload("r1", "00:03"),
        ],
    )

    code = main([], events_file=events_file)

    out = capsys.readouterr().out
    assert code == 0
    assert "AI AGENT" in out
    assert "Runs completed" in out
    assert "Total tokens" in out
    assert "Estimated cost" in out
    assert "Average run" in out
    assert "Cache hit rate" in out
    assert "Most expensive tools:" in out
    assert "exec:git" in out


def test_empty_storage_prints_zero_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([], events_file=tmp_path / "absent.jsonl")

    out = capsys.readouterr().out
    assert code == 0
    assert "AI AGENT" in out
    assert "Runs completed" in out


def test_task_flag_prints_timeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events_file = write_events(
        tmp_path / "events.jsonl",
        [
            llm_payload("r1", 1, "00:01"),
            tool_payload("r1", "exec:git", "00:02"),
            run_payload("r1", "00:03"),
        ],
    )

    code = main(["--task", "r1"], events_file=events_file)

    out = capsys.readouterr().out
    assert code == 0
    assert "Run r1" in out
    assert "Step 1" in out
    assert "1,000 tokens" in out
    assert "exec:git" in out


def test_unknown_task_fails_with_message(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events_file = write_events(tmp_path / "events.jsonl", [run_payload("r1", "00:03")])

    code = main(["--task", "nope"], events_file=events_file)

    captured = capsys.readouterr()
    assert code == 1
    assert "nope" in captured.err
    assert captured.out == ""
