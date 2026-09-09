"""Контракт инструмента агентного цикла."""

from typing import Protocol

from bot.agent.progress import AgentProgress
from bot.domain.ids import RequestId
from bot.domain.tools import ExecutionContext, ToolCall, ToolResult, ToolSpec


class Tool(Protocol):
    """Инструмент: описание для модели плюс выполнение вызова.

    ``context`` — контекст выполнения запуска (скоуп владельца и т.п.):
    цикл прокидывает его каждому инструменту, тот решает сам, нужен ли он.
    """

    @property
    def spec(self) -> ToolSpec: ...

    async def execute(
        self,
        request_id: RequestId,
        call: ToolCall,
        progress: AgentProgress,
        context: ExecutionContext,
    ) -> ToolResult: ...
