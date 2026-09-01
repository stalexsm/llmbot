"""Декоратор на шве Tool: событие tool_call на каждый вызов exec."""

import json
from pathlib import Path

import pytest
import structlog.stdlib

from bot.agent.progress import AgentProgress, NullProgress
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ToolCall, ToolResult, ToolSpec
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.recorder import MetricsRecorder
from bot.metrics.tool import MeteredTool

REQUEST_ID = RequestId("req-1")
PROGRESS = NullProgress()


class StubTool:
    """Фальшивка Tool: отдаёт заготовленный результат, shell не запускает."""

    def __init__(self, result: ToolResult) -> None:
        self._result = result
        self.spec = ToolSpec(
            name=ToolId("execute_command"),
            description="stub",
            parameters=(),
            required=(),
        )

    async def execute(
        self, request_id: RequestId, call: ToolCall, progress: AgentProgress
    ) -> ToolResult:
        return self._result


def make_collector(tmp_path: Path) -> tuple[RunMetricsCollector, Path]:
    directory = tmp_path / "metrics"
    collector = RunMetricsCollector(
        recorder=MetricsRecorder(directory=directory, logger=structlog.stdlib.get_logger()),
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
    )
    return collector, directory


async def run_metered(
    tmp_path: Path, arguments: str, result: ToolResult
) -> tuple[dict, ToolResult]:
    """Один вызов обёрнутого инструмента; возвращает JSONL-строку и результат."""
    collector, directory = make_collector(tmp_path)
    inner = StubTool(result)
    metered = MeteredTool(inner=inner, collector=collector)
    returned = await metered.execute(
        REQUEST_ID,
        ToolCall(name=ToolId("execute_command"), arguments=arguments),
        PROGRESS,
    )
    line = json.loads((directory / "events.jsonl").read_text(encoding="utf-8"))
    return line, returned


async def test_tool_call_recorded_with_class_sizes_and_status(tmp_path: Path) -> None:
    arguments = json.dumps({"command": "git status"}, ensure_ascii=False)
    content = "exit_code: 0\nstdout:\nmain\n"
    line, returned = await run_metered(
        tmp_path, arguments, ToolResult(content=content, succeeded=True)
    )

    assert line["kind"] == "tool_call"
    assert line["request_id"] == "req-1"
    assert line["tool_name"] == "exec:git"
    assert line["input_size"] == len(arguments)
    assert line["output_size"] == len(content)
    assert line["output_tokens"] == len(content) // 4
    assert line["duration_ms"] >= 0
    assert line["succeeded"] is True
    # Результат проходит насквозь без изменений.
    assert returned.content == content
    assert "timestamp" in line


async def test_command_and_output_content_not_recorded(tmp_path: Path) -> None:
    arguments = json.dumps({"command": "cat .env"}, ensure_ascii=False)
    content = "exit_code: 0\nstdout:\nTELEGRAM_BOT_TOKEN=секрет\n"
    line, _ = await run_metered(tmp_path, arguments, ToolResult(content=content, succeeded=True))

    dumped = json.dumps(line, ensure_ascii=False)
    assert "cat .env" not in dumped
    assert "TELEGRAM_BOT_TOKEN" not in dumped
    assert "секрет" not in dumped


@pytest.mark.parametrize(
    ("command", "expected_tool_name"),
    [
        ("uv run pytest -q", "exec:python"),
        ("rg pattern .", "exec:rg"),
        ("sed -n 1p f.txt", "exec:cat"),
        ("ls -la", "exec:ls"),
        ("curl https://example.com", "exec:other"),
    ],
)
async def test_command_classes_map_to_tool_name(
    tmp_path: Path, command: str, expected_tool_name: str
) -> None:
    line, _ = await run_metered(
        tmp_path,
        json.dumps({"command": command}, ensure_ascii=False),
        ToolResult(content="x", succeeded=True),
    )
    assert line["tool_name"] == expected_tool_name


async def test_invalid_arguments_recorded_as_other_failure(tmp_path: Path) -> None:
    line, _ = await run_metered(
        tmp_path,
        "не json",
        ToolResult(content="invalid tool arguments", succeeded=False),
    )
    assert line["tool_name"] == "exec:other"
    assert line["succeeded"] is False
    assert line["input_size"] == len("не json")


def test_spec_is_delegated(tmp_path: Path) -> None:
    collector, _ = make_collector(tmp_path)
    inner = StubTool(ToolResult(content="", succeeded=True))
    assert MeteredTool(inner=inner, collector=collector).spec is inner.spec
