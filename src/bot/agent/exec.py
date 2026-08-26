"""Инструмент exec: выполнение консольной команды в shell на машине агента.

Одна строка команды, рабочий каталог задаётся при сборке (корень проекта),
зависшая команда обрывается по таймауту. Модель получает exit code, stdout
и stderr одним текстом, суммарно обрезанным до лимита символов.
"""

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path

import structlog

from bot.agent.progress import AgentProgress
from bot.domain.ids import ToolId
from bot.domain.tools import ToolCall, ToolParameter, ToolResult, ToolSpec

_TRUNCATION_MARKER = "\n…[вывод обрезан]"


def _truncate(text: str, limit: int) -> str:
    """Обрезать текст до лимита, оставив маркер обрезки в пределах лимита."""
    if len(text) <= limit:
        return text
    keep = max(limit - len(_TRUNCATION_MARKER), 0)
    return text[:keep] + _TRUNCATION_MARKER


class ExecTool:
    """Единственный инструмент агента: shell-команда с таймаутом и обрезкой вывода."""

    def __init__(
        self,
        cwd: Path,
        timeout_seconds: float,
        max_output_chars: int,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars
        self._logger = logger.bind(component="exec_tool")
        self._spec = ToolSpec(
            name=ToolId("exec"),
            description=(
                "Выполнить консольную команду в shell на машине агента (одна строка). "
                "Возвращает exit code, stdout и stderr."
            ),
            parameters=(
                ToolParameter(
                    name="command",
                    type="string",
                    description="Команда для выполнения, одна строка shell",
                ),
            ),
            required=("command",),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall, progress: AgentProgress) -> ToolResult:
        command = self._extract_command(call)
        if command is None:
            return ToolResult(
                content=_truncate(
                    "invalid tool arguments: expected JSON object with a string field "
                    f'"command", got: {call.arguments}',
                    self._max_output_chars,
                ),
                succeeded=False,
            )

        await self._report_started(progress, command)
        started_at = time.monotonic()
        result = await self._run(command)
        await self._report_finished(progress, command, result.succeeded)
        self._logger.info(
            "command_executed",
            duration_ms=int((time.monotonic() - started_at) * 1000),
            succeeded=result.succeeded,
        )
        return result

    async def _run(self, command: str) -> ToolResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            await self._kill(process)
            content = (
                f"exit_code: timeout (limit {self._timeout_seconds} s)\n"
                "stdout:\n\nstderr:\nкоманда прервана: превышен таймаут"
            )
            return ToolResult(content=_truncate(content, self._max_output_chars), succeeded=False)
        output_text = (
            f"exit_code: {process.returncode}\n"
            f"stdout:\n{stdout.decode(errors='replace')}\n"
            f"stderr:\n{stderr.decode(errors='replace')}"
        )
        return ToolResult(
            content=_truncate(output_text, self._max_output_chars),
            succeeded=process.returncode == 0,
        )

    async def _kill(self, process: asyncio.subprocess.Process) -> None:
        """Убить процесс (и его группу), чтобы зависшая команда не пережила таймаут."""
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            await process.wait()

    @staticmethod
    def _extract_command(call: ToolCall) -> str | None:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        return command

    async def _report_started(self, progress: AgentProgress, command: str) -> None:
        # Сбой показа шага в чате не должен рывать выполнение команды.
        try:
            await progress.command_started(command)
        except Exception:
            self._logger.warning("progress_report_failed", phase="started")

    async def _report_finished(
        self, progress: AgentProgress, command: str, succeeded: bool
    ) -> None:
        try:
            await progress.command_finished(command, succeeded)
        except Exception:
            self._logger.warning("progress_report_failed", phase="finished")
