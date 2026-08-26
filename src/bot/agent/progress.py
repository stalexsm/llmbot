"""Прогресс агентного цикла, видимый пользователю в чате.

Перед выполнением команды приходит сообщение с командой, после выполнения
оно редактируется в отметку об успехе. Реализация протокола принадлежит
Telegram-слою; агентный слой только вызывает его.
"""

from typing import Protocol


class AgentProgress(Protocol):
    """Sink для шагов агента, наблюдаемых пользователем."""

    async def command_started(self, command: str) -> None: ...

    async def command_finished(self, command: str, succeeded: bool) -> None: ...


class NullProgress:
    """Прогресс по умолчанию: шаги не показываются."""

    async def command_started(self, command: str) -> None:
        return None

    async def command_finished(self, command: str, succeeded: bool) -> None:
        return None
