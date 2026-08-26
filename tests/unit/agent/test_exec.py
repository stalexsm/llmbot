"""Unit tests for the exec tool: shell execution with timeout and output trimming."""

import json
import os
from pathlib import Path

import pytest
import structlog.stdlib

from bot.agent.exec import ExecTool
from bot.agent.progress import NullProgress
from bot.domain.ids import ToolId
from bot.domain.tools import ToolCall


class RecordingProgress:
    """Progress double: фиксирует вызовы в порядке поступления."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, bool | None]] = []

    async def command_started(self, command: str) -> None:
        self.events.append(("started", command, None))

    async def command_finished(self, command: str, succeeded: bool) -> None:
        self.events.append(("finished", command, succeeded))


class FailingProgress:
    """Progress double: падает — сбои показа шагов не должны рвать выполнение."""

    async def command_started(self, command: str) -> None:
        raise RuntimeError("telegram is down")

    async def command_finished(self, command: str, succeeded: bool) -> None:
        raise RuntimeError("telegram is down")


def make_tool(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    *,
    timeout: float = 10.0,
    max_output_chars: int = 4000,
) -> ExecTool:
    return ExecTool(
        cwd=tmp_path,
        timeout_seconds=timeout,
        max_output_chars=max_output_chars,
        logger=logger,
    )


def exec_call(command: str) -> ToolCall:
    return ToolCall(name=ToolId("exec"), arguments=json.dumps({"command": command}))


def realpath(path: str) -> str:
    """Синхронный helper: ASYNC240 запрещает pathlib/os.path прямо в async-тестах."""
    return os.path.realpath(path)


async def test_successful_command_returns_stdout_and_exit_code(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(exec_call("echo привет"), NullProgress())

    assert result.succeeded is True
    assert "exit_code: 0" in result.content
    assert "stdout:" in result.content
    assert "привет" in result.content


async def test_failing_command_is_visible_to_the_model(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(exec_call("echo беда >&2; exit 3"), NullProgress())

    assert result.succeeded is False
    assert "exit_code: 3" in result.content
    assert "беда" in result.content
    assert "stderr:" in result.content


async def test_working_directory_is_tool_argument(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(exec_call("pwd"), NullProgress())

    # На macOS tmp_path указывает через симлинк /private — сравниваем realpath.
    printed = result.content.splitlines()[2].strip()
    assert realpath(printed) == realpath(str(tmp_path))


async def test_hung_command_is_aborted_by_timeout(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger, timeout=0.2)

    result = await tool.execute(exec_call("sleep 30"), NullProgress())

    assert result.succeeded is False
    assert "timeout" in result.content


async def test_long_output_is_trimmed_to_char_limit(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger, max_output_chars=500)

    result = await tool.execute(exec_call("python3 -c \"print('a' * 5000)\""), NullProgress())

    assert len(result.content) <= 515  # лимит плюс маркер обрезки
    assert "вывод обрезан" in result.content
    assert result.content.count("a") < 5000


async def test_invalid_json_arguments_return_error_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(ToolCall(name=ToolId("exec"), arguments="не json"), NullProgress())

    assert result.succeeded is False
    assert "command" in result.content  # подсказка модели, что ожидается


async def test_missing_command_argument_returns_error_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(ToolCall(name=ToolId("exec"), arguments="{}"), NullProgress())

    assert result.succeeded is False
    assert "command" in result.content


async def test_progress_receives_started_then_finished(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)
    progress = RecordingProgress()

    result = await tool.execute(exec_call("echo ок"), progress)

    assert result.succeeded is True
    assert progress.events == [
        ("started", "echo ок", None),
        ("finished", "echo ок", True),
    ]


async def test_progress_failure_does_not_break_execution(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(exec_call("echo ок"), FailingProgress())

    assert result.succeeded is True
    assert "ок" in result.content


def test_spec_describes_exec_tool(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    tool = make_tool(tmp_path, logger)

    spec = tool.spec

    assert spec.name == ToolId("exec")
    assert spec.required == ("command",)
    assert [param.name for param in spec.parameters] == ["command"]
    assert all(param.type == "string" for param in spec.parameters)


@pytest.mark.parametrize("command", ["echo ok", "pwd"])
async def test_result_content_starts_with_exit_code(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger, command: str
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(exec_call(command), NullProgress())

    assert result.content.startswith("exit_code:")
