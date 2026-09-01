"""Декоратор на шве Tool: событие tool_call на каждый вызов инструмента.

Реализует Protocol ``Tool`` структурно и подключается в композиционном корне
поверх исполнителя команд. Замеряет длительность и размеры ввода/вывода,
классифицирует команду и передаёт всё в ``RunMetricsCollector``; содержимое
команды и её вывода в метрики не попадает.
"""

import time

from bot.agent.command_class import classify_call_arguments
from bot.agent.progress import AgentProgress
from bot.agent.tools import Tool
from bot.domain.ids import RequestId
from bot.domain.tools import ToolCall, ToolResult, ToolSpec
from bot.metrics.collector import RunMetricsCollector


class MeteredTool:
    """Замеряет и учитывает каждый вызов инструмента поверх внутреннего исполнителя."""

    def __init__(self, inner: Tool, collector: RunMetricsCollector) -> None:
        self._inner = inner
        self._collector = collector

    @property
    def spec(self) -> ToolSpec:
        return self._inner.spec

    async def execute(
        self, request_id: RequestId, call: ToolCall, progress: AgentProgress
    ) -> ToolResult:
        started_at = time.monotonic()
        result = await self._inner.execute(request_id, call, progress)
        # Метка класса: exec:<класс> — git, python, rg, cat, ls или other.
        # Размер ввода — сырые аргументы вызова (то, что породила модель),
        # размера вывода — текст результата; сами тексты не записываются.
        self._collector.record_tool_call(
            request_id=request_id,
            tool_name=f"exec:{classify_call_arguments(call.arguments).value}",
            input_size=len(call.arguments),
            output_size=len(result.content),
            duration_ms=int((time.monotonic() - started_at) * 1000),
            succeeded=result.succeeded,
        )
        return result
