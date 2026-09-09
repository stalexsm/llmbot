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
from bot.domain.ids import RequestId, ToolId
from bot.domain.tools import ExecutionContext, ToolCall, ToolResult, ToolSpec
from bot.metrics.collector import RunMetricsCollector

# Имя exec-инструмента: его вызовы в метриках маркируются классом команды
# (exec:<класс>), остальные инструменты — собственным именем спеки.
_EXEC_TOOL_NAME = ToolId("execute_command")


class MeteredTool:
    """Замеряет и учитывает каждый вызов инструмента поверх внутреннего исполнителя."""

    def __init__(self, inner: Tool, collector: RunMetricsCollector) -> None:
        self._inner = inner
        self._collector = collector

    @property
    def spec(self) -> ToolSpec:
        return self._inner.spec

    async def execute(
        self,
        request_id: RequestId,
        call: ToolCall,
        progress: AgentProgress,
        context: ExecutionContext,
    ) -> ToolResult:
        started_at = time.monotonic()
        try:
            result = await self._inner.execute(request_id, call, progress, context)
        except Exception:
            # Неудачный вызов — тоже вызов: пишем событие со сбоем (вывода нет)
            # и пробрасываем исключение дальше, как его отдал исполнитель, —
            # симметрично MeteredInferenceProvider («включая неудачный»).
            self._record(request_id, call, started_at, output_size=0, succeeded=False)
            raise
        self._record(
            request_id,
            call,
            started_at,
            output_size=len(result.content),
            succeeded=result.succeeded,
        )
        return result

    def _record(
        self,
        request_id: RequestId,
        call: ToolCall,
        started_at: float,
        *,
        output_size: int,
        succeeded: bool,
    ) -> None:
        # Метка класса: exec:<класс> — git, python, rg, cat, ls или other;
        # прочие инструменты учитываются под именем своей спеки (search_documents).
        # Размер ввода — сырые аргументы вызова (то, что породила модель),
        # размер вывода — модель-видимый текст результата (после обрезки exec);
        # сами тексты не записываются.
        if self._inner.spec.name == _EXEC_TOOL_NAME:
            tool_name = f"exec:{classify_call_arguments(call.arguments).value}"
        else:
            tool_name = str(self._inner.spec.name)
        self._collector.record_tool_call(
            request_id=request_id,
            tool_name=tool_name,
            input_size=len(call.arguments),
            output_size=output_size,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            succeeded=succeeded,
        )
