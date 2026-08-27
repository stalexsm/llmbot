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
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ToolCall, ToolParameter, ToolResult, ToolSpec

_TRUNCATION_MARKER = "\n…[вывод обрезан]"

# Дренаж потоков после убийства группы: обычно мгновенный (EOF приходит
# сразу), но потомок, сбежавший из группы через новую сессию, может держать
# пайпы открытыми вечно. Короткая верхняя граница гарантирует, что после
# таймаута execute() вернёт результат, а не заморозит чат навсегда.
_POST_KILL_DRAIN_SECONDS: float = 5.0

# Подсказки самокоррекции для типовых shell-ошибок: результат инструмента —
# единственный канал обратной связи для модели, а крошечные модели регулярно
# выдумывают несуществующие программы (например «weather» вместо чтения скилла).
# Подсказка не отменяет факты (exit code, stderr остаются как есть), она лишь
# подсказывает причину и правильный следующий шаг.
_SELF_CORRECTION_HINTS: dict[int, str] = {
    126: (
        "\n\nhint: оболочка не смогла исполнить команду (файл не является программой). "
        "Вероятно, имя команды или её аргументы выдуманы. exec запускает только реальные "
        "утилиты (cat, ls, curl, …); файлы читаются командой cat. Если задача подходит под "
        "скилл из индекса, начни с команды: cat skills/<имя>/SKILL.md"
    ),
    127: (
        "\n\nhint: такой программы нет на машине (command not found). Не выдумывай имена "
        "команд: exec запускает только реальные утилиты (cat, ls, curl, …). Если задача "
        "подходит под скилл из индекса, начни с команды: cat skills/<имя>/SKILL.md"
    ),
}


def _self_correction_hint(returncode: int | None, stdout: str = "", stderr: str = "") -> str:
    """Подсказка модели для типовых сбоев; пустая строка — без подсказки."""
    hint = _SELF_CORRECTION_HINTS.get(returncode, "") if returncode is not None else ""
    if hint:
        return hint
    if returncode == 0:
        lowered = f"{stdout}\n{stderr}".lower()
        if (
            "403" in lowered
            or "401" in lowered
            or "forbidden" in lowered
            or "unauthorized" in lowered
        ):
            # Команда «успешна» (exit 0), но сервис отдал страницу-отказ доступа:
            # API с ключом, выдумывать который нельзя.
            hint = (
                "\n\nhint: сервис ответил отказом доступа (403/401): этому API нужен "
                "платный ключ. Не выдумывай API с ключами — используй бесплатный "
                "источник без авторизации из файла скилла: cat skills/<имя>/SKILL.md"
            )
    return hint


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
            name=ToolId("execute_command"),
            description=(
                "Выполнить одну строку shell в рабочем каталоге проекта; возвращает "
                "exit code, stdout и stderr. Для задач с живыми данными сначала прочитай "
                "файл подходящего скилла командой cat skills/<имя>/SKILL.md и следуй ему."
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

    async def execute(
        self, request_id: RequestId, call: ToolCall, progress: AgentProgress
    ) -> ToolResult:
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
            request_id=request_id,
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
        communicate_task = asyncio.create_task(process.communicate())
        stdout = stderr = b""
        timed_out = False
        try:
            # shield: таймаут отменяет ожидание, но не само чтение — иначе
            # уже вычитанный вывод пропадёт вместе с задачей.
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=self._timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            await self._kill(process)
            # Группа убита — потоки закрываются, и та же задача докатывается
            # до конца, отдавая и частичный вывод для следующей попытки модели.
            # Сбежавший из группы потомок (новая сессия) держит пайпы открытыми
            # и EOF не придёт, поэтому дренаж ограничен коротким таймаутом:
            # execute() обязан вернуться, чтобы чат не замер вместе с блокировкой.
            try:
                stdout, stderr = await asyncio.wait_for(
                    communicate_task, timeout=_POST_KILL_DRAIN_SECONDS
                )
            except Exception:
                # Частичный вывод теряется вместе с отменённой задачей — сознательная
                # плата за гарантированный возврат. Транспорты пайпов закрываем, чтобы
                # не оставлять открытых дескрипторов позади отменённой задачи: публичного
                # способа закрыть их у asyncio нет, читаем внутренний хэндл защищённо.
                transport = getattr(process, "_transport", None)
                if transport is not None:
                    with contextlib.suppress(Exception):
                        transport.close()
        if timed_out:
            content = (
                f"exit_code: timeout (limit {self._timeout_seconds} s)\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )
            return ToolResult(content=_truncate(content, self._max_output_chars), succeeded=False)
        output_text = (
            f"exit_code: {process.returncode}\n"
            f"stdout:\n{stdout.decode(errors='replace')}\n"
            f"stderr:\n{stderr.decode(errors='replace')}"
            + _self_correction_hint(
                process.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
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
