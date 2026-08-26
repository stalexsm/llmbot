"""Агентный цикл: «ответ модели → выполнение инструментов → следующий запрос».

Шаг — один ответ модели. Ответ с вызовами инструментов продолжает цикл,
ответ без них — финальный и завершает его. Лимит шагов защищает от
зацикливания: при исчерпании цикл останавливается честно, не выдумывая
финальный ответ.
"""

import time
from dataclasses import dataclass

import structlog

from bot.agent.progress import AgentProgress, NullProgress
from bot.agent.tools import Tool
from bot.application.errors import EmptyInferenceResponseError
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall, ToolResult
from bot.inference.models import InferenceRequest
from bot.inference.provider import InferenceProvider

_NULL_PROGRESS = NullProgress()


@dataclass(frozen=True)
class AgentRun:
    """Результат одного агентного цикла.

    ``exchange`` — все сообщения обмена (включая сообщение пользователя),
    которые дописываются в чат-сессию: пользователь, пары «вызов
    инструмента → результат», финальный ответ модели.
    """

    steps_used: int
    exchange: tuple[InferenceMessage, ...]
    final_answer: str | None
    stopped_by_limit: bool


class AgentLoop:
    """Крутит шаги до финального ответа модели или лимита шагов."""

    def __init__(
        self,
        inference: InferenceProvider,
        model: ModelId,
        system_prompt: str,
        tools: tuple[Tool, ...],
        step_limit: int,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._inference = inference
        self._model = model
        self._system_prompt = system_prompt
        self._step_limit = step_limit
        self._logger = logger.bind(component="agent_loop")
        self._tools = {tool.spec.name: tool for tool in tools}
        self._tool_specs = tuple(tool.spec for tool in tools)

    async def run(
        self,
        request_id: RequestId,
        history: tuple[InferenceMessage, ...],
        user_message: InferenceMessage,
        progress: AgentProgress | None = None,
    ) -> AgentRun:
        reporter = progress if progress is not None else _NULL_PROGRESS
        started_at = time.monotonic()
        messages: list[InferenceMessage] = [
            InferenceMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
            *history,
            user_message,
        ]
        exchange: list[InferenceMessage] = [user_message]
        for step in range(1, self._step_limit + 1):
            response = await self._inference.generate(
                InferenceRequest(
                    request_id=request_id,
                    model=self._model,
                    messages=tuple(messages),
                    tools=self._tool_specs,
                )
            )
            assistant = InferenceMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            if not response.tool_calls:
                if not response.content.strip():
                    self._logger.warning(
                        "agent_step_empty_final",
                        request_id=request_id,
                        step=step,
                    )
                    raise EmptyInferenceResponseError(
                        "Inference provider returned an empty response"
                    )
                exchange.append(assistant)
                self._log_run(request_id, started_at, steps=step, stopped=False)
                return AgentRun(
                    steps_used=step,
                    exchange=tuple(exchange),
                    final_answer=response.content,
                    stopped_by_limit=False,
                )
            messages.append(assistant)
            exchange.append(assistant)
            for call in response.tool_calls:
                result = await self._execute(request_id, call, reporter)
                result_message = InferenceMessage(
                    role=MessageRole.TOOL,
                    content=result.content,
                    tool_name=call.name,
                )
                messages.append(result_message)
                exchange.append(result_message)
            self._logger.info(
                "agent_step_finished",
                request_id=request_id,
                step=step,
                tool_calls=len(response.tool_calls),
            )
        self._log_run(request_id, started_at, steps=self._step_limit, stopped=True)
        return AgentRun(
            steps_used=self._step_limit,
            exchange=tuple(exchange),
            final_answer=None,
            stopped_by_limit=True,
        )

    async def _execute(
        self,
        request_id: RequestId,
        call: ToolCall,
        progress: AgentProgress,
    ) -> ToolResult:
        """Выполнить вызов инструмента; неизвестное имя не рвёт цикл."""
        tool = self._tools.get(call.name)
        if tool is None:
            self._logger.warning(
                "agent_unknown_tool",
                request_id=request_id,
                tool=call.name,
            )
            return ToolResult(
                content=f"unknown tool: {call.name}. Доступные инструменты: "
                f"{', '.join(str(name) for name in self._tools)}.",
                succeeded=False,
            )
        return await tool.execute(request_id, call, progress)

    def _log_run(
        self,
        request_id: RequestId,
        started_at: float,
        *,
        steps: int,
        stopped: bool,
    ) -> None:
        self._logger.info(
            "agent_run_finished",
            request_id=request_id,
            model=self._model,
            steps=steps,
            stopped_by_limit=stopped,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            status="stopped_by_limit" if stopped else "success",
        )
