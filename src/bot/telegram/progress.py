"""Отображение шагов агентного цикла в Telegram-чате.

Перед выполнением команды отправляется сообщение с командой, после
выполнения оно редактируется в отметку об успехе или неудаче.
"""

import structlog
from aiogram.types import Message

_RUNNING_PREFIX = "⏳"
_SUCCESS_MARK = "✅"
_FAILURE_MARK = "❌"
# Команда может быть очень длинной; Telegram ограничивает сообщение 4096 символами.
_MAX_COMMAND_DISPLAY_CHARS = 400


class TelegramCommandProgress:
    """Реализация ``AgentProgress`` поверх Telegram-сообщений."""

    def __init__(self, source: Message, logger: structlog.stdlib.BoundLogger) -> None:
        self._source = source
        self._logger = logger.bind(component="telegram_command_progress")
        self._status: Message | None = None

    async def command_started(self, command: str) -> None:
        display = self._display(command)
        self._status = await self._source.answer(f"{_RUNNING_PREFIX} {display}")

    async def command_finished(self, command: str, succeeded: bool) -> None:
        display = self._display(command)
        mark = _SUCCESS_MARK if succeeded else _FAILURE_MARK
        text = f"{mark} {display}"
        if self._status is not None:
            await self._status.edit_text(text)
        else:
            # Сообщение-статус не отправилось (сбой на старте шага) — падаем в обычное.
            self._status = await self._source.answer(text)

    @staticmethod
    def _display(command: str) -> str:
        if len(command) <= _MAX_COMMAND_DISPLAY_CHARS:
            return command
        return command[:_MAX_COMMAND_DISPLAY_CHARS] + "…"
