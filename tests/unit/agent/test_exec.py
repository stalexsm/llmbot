"""Unit tests for the exec tool: shell execution with timeout and output trimming."""

import asyncio
import json
import os
from pathlib import Path

import pytest
import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.agent import exec as exec_module
from bot.agent.exec import ExecTool
from bot.agent.progress import NullProgress
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ExecutionContext, ToolCall
from tests.fakes import RecordingProgress


class FailingProgress:
    """Progress double: падает — сбои показа шагов не должны рвать выполнение."""

    async def command_started(self, command: str) -> None:
        raise RuntimeError("telegram is down")

    async def command_finished(self, command: str, succeeded: bool) -> None:
        raise RuntimeError("telegram is down")


REQUEST_ID = RequestId("exec-test-request")
# ExecTool контекст выполнения (скоуп владельца) игнорирует — передаём заглушку.
_CONTEXT = ExecutionContext()


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
    return ToolCall(name=ToolId("execute_command"), arguments=json.dumps({"command": command}))


def realpath(path: str) -> str:
    """Синхронный helper: ASYNC240 запрещает pathlib/os.path прямо в async-тестах."""
    return os.path.realpath(path)


async def test_successful_command_returns_stdout_and_exit_code(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call("echo привет"), NullProgress(), _CONTEXT)

    assert result.succeeded is True
    assert "exit_code: 0" in result.content
    assert "stdout:" in result.content
    assert "привет" in result.content


async def test_failing_command_is_visible_to_the_model(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID, exec_call("echo беда >&2; exit 3"), NullProgress(), _CONTEXT
    )

    assert result.succeeded is False
    assert "exit_code: 3" in result.content
    assert "беда" in result.content
    assert "stderr:" in result.content


async def test_command_not_found_gets_self_correction_hint(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID, exec_call("definitely-not-a-real-cmd"), NullProgress(), _CONTEXT
    )

    assert result.succeeded is False
    assert result.content.startswith("exit_code: 127")
    assert "hint:" in result.content
    assert "command not found" in result.content or "нет на машине" in result.content


async def test_permission_denied_gets_self_correction_hint(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    (tmp_path / "not-executable.txt").write_text("данные", encoding="utf-8")
    tool = make_tool(tmp_path, logger)

    # Файл существует, но не исполняем — классическая ошибка 126.
    result = await tool.execute(
        REQUEST_ID, exec_call("./not-executable.txt"), NullProgress(), _CONTEXT
    )

    assert result.succeeded is False
    assert result.content.startswith("exit_code: 126")
    assert "hint:" in result.content


async def test_successful_command_has_no_hint(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call("echo ok"), NullProgress(), _CONTEXT)

    assert "hint:" not in result.content


async def test_http_access_denied_in_output_gets_hint(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    """Exit 0 при странице-отказе API (403) — команда «успешна», но данных нет."""
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID,
        exec_call("echo '<html><title>403 Forbidden</title></html>'"),
        NullProgress(),
        _CONTEXT,
    )

    assert result.succeeded is True
    assert result.content.startswith("exit_code: 0")
    assert "hint:" in result.content


async def test_working_directory_is_tool_argument(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call("pwd"), NullProgress(), _CONTEXT)

    # На macOS tmp_path указывает через симлинк /private — сравниваем realpath.
    printed = result.content.splitlines()[2].strip()
    assert realpath(printed) == realpath(str(tmp_path))


async def test_hung_command_is_aborted_by_timeout(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger, timeout=0.2)

    result = await tool.execute(REQUEST_ID, exec_call("sleep 30"), NullProgress(), _CONTEXT)

    assert result.succeeded is False
    assert "timeout" in result.content


async def test_execute_returns_after_process_group_escape(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбежавший из группы потомок держит пайпы: execute() всё равно возвращается.

    Регрессия: killpg не берёт потомка, ушедшего в новую сессию, его копии
    stdout/stderr не дают EOF, и communicate ждал вечно — чат замерал вместе
    с блокировкой сессии. Дренаж после kill ограничен отдельным таймаутом.
    """
    monkeypatch.setattr(exec_module, "_POST_KILL_DRAIN_SECONDS", 0.2)
    tool = make_tool(tmp_path, logger, timeout=0.3)
    # Внутренний sh уходит сразу после echo, а python3 перед сном уходит в новую
    # сессию (os.setsid) и держит унаследованные пайпы: EOF после killpg не придет.
    command = "sh -c 'python3 -c \"import os, time; os.setsid(); time.sleep(2)\" & echo started'"

    result = await asyncio.wait_for(
        tool.execute(REQUEST_ID, exec_call(command), NullProgress(), _CONTEXT), timeout=10.0
    )

    assert result.succeeded is False
    assert "timeout" in result.content


async def test_timed_out_command_keeps_partial_output(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger, timeout=0.5)

    result = await tool.execute(
        REQUEST_ID, exec_call("echo частичный-вывод; sleep 30"), NullProgress(), _CONTEXT
    )

    assert result.succeeded is False
    assert "timeout" in result.content
    assert "частичный-вывод" in result.content


async def test_long_output_is_trimmed_to_char_limit(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger, max_output_chars=500)

    result = await tool.execute(
        REQUEST_ID, exec_call("python3 -c \"print('a' * 5000)\""), NullProgress(), _CONTEXT
    )

    assert len(result.content) <= 500  # прежний лимит символов соблюдён
    assert result.content.startswith("exit_code: 0")  # голова вывода сохранена
    assert "stderr:" in result.content  # хвост вывода сохранён
    assert "пропущено" in result.content  # маркер обрезки понятен модели
    assert result.content.count("a") < 5000


NOISY_SCRIPT = (
    "import sys\n"
    "for frame in ('\\x1b[32mскачивание\\x1b[0m 10%', '⠋ 50%', '\\x1b[1m⠹ 100%\\x1b[0m'):\n"
    "    sys.stdout.write(frame + '\\r')\n"
    "sys.stdout.write('\\r\\nГотово\\n')\n"
)


async def test_noisy_real_command_output_is_cleaned(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    """Реальная команда с шумным выводом: модель видит финальное состояние."""
    (tmp_path / "noisy.py").write_text(NOISY_SCRIPT, encoding="utf-8")
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call("python3 noisy.py"), NullProgress(), _CONTEXT)

    assert result.succeeded is True
    assert "\x1b" not in result.content  # ANSI-коды сняты
    assert "\r" not in result.content  # кадры прогресса схлопнуты
    assert "10%" not in result.content and "⠋" not in result.content  # промежуточные кадры
    assert "100%" in result.content  # финальный кадр сохранён
    assert "Готово" in result.content


async def test_invalid_json_arguments_return_error_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID,
        ToolCall(name=ToolId("execute_command"), arguments="не json"),
        NullProgress(),
        _CONTEXT,
    )

    assert result.succeeded is False
    assert "command" in result.content  # подсказка модели, что ожидается


async def test_missing_command_argument_returns_error_result(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(
        REQUEST_ID,
        ToolCall(name=ToolId("execute_command"), arguments="{}"),
        NullProgress(),
        _CONTEXT,
    )

    assert result.succeeded is False
    assert "command" in result.content


async def test_command_event_is_logged_with_request_id(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger, capturing_logger: CapturingLogger
) -> None:
    # §19: request_id в событиях — корреляция exec-шага с запросом.
    tool = make_tool(tmp_path, logger)

    await tool.execute(REQUEST_ID, exec_call("echo ок"), NullProgress(), _CONTEXT)

    events = [call.kwargs for call in capturing_logger.calls if call.kwargs.get("event")]
    executed = [kwargs for kwargs in events if kwargs.get("event") == "command_executed"]
    assert executed and executed[0].get("request_id") == REQUEST_ID


async def test_progress_receives_started_then_finished(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)
    progress = RecordingProgress()

    result = await tool.execute(REQUEST_ID, exec_call("echo ок"), progress, _CONTEXT)

    assert result.succeeded is True
    assert progress.events == [
        ("started", "echo ок", None),
        ("finished", "echo ок", True),
    ]


async def test_progress_failure_does_not_break_execution(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call("echo ок"), FailingProgress(), _CONTEXT)

    assert result.succeeded is True
    assert "ок" in result.content


def test_spec_describes_exec_tool(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    tool = make_tool(tmp_path, logger)

    spec = tool.spec

    assert spec.name == ToolId("execute_command")
    assert spec.required == ("command",)
    assert [param.name for param in spec.parameters] == ["command"]
    assert all(param.type == "string" for param in spec.parameters)


@pytest.mark.parametrize("command", ["echo ok", "pwd"])
async def test_result_content_starts_with_exit_code(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger, command: str
) -> None:
    tool = make_tool(tmp_path, logger)

    result = await tool.execute(REQUEST_ID, exec_call(command), NullProgress(), _CONTEXT)

    assert result.content.startswith("exit_code:")
