"""Агентный цикл: «ответ модели → выполнение инструментов → следующий запрос».

Шаг — один ответ модели. Ответ с вызовами инструментов продолжает цикл,
ответ без них — финальный и завершает его. Лимит шагов защищает от
зацикливания: при исчерпании цикл останавливается честно, не выдумывая
финальный ответ.
"""

import time
from dataclasses import dataclass, replace

import structlog

from bot.agent.command_class import command_from_arguments
from bot.agent.progress import AgentProgress, NullProgress
from bot.agent.prompts import build_date_block
from bot.agent.tools import Tool
from bot.application.errors import EmptyInferenceResponseError
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall, ToolResult
from bot.inference.models import InferenceRequest
from bot.inference.provider import InferenceProvider

_NULL_PROGRESS = NullProgress()

# Сколько раз повторять шаг при пустом финальном ответе прежде чем рвать цикл:
# крошечные reasoning-модели эпизодически выдают пустоту даже в режиме thinking,
# повтор того же запроса обычно даёт содержательный ответ.
_EMPTY_STEP_RETRIES = 2


def _strip_tool_echo(content: str, tool_results: tuple[str, ...]) -> str:
    """Убрать из финального ответа дословную копию результатов инструментов.

    Крошечные модели иногда переписывают сырой результат инструмента
    (exit_code, stdout, …) прямо в финальный ответ, нарушая правило промпта.
    Эхо распознаётся по строкам самих результатов: блок начинается со строки,
    с которой начинается какой-то результат, и продолжается, пока строки
    дословно встречаются в результатах (пустые строки внутри блока — тоже его
    часть). Собственный текст модели после блока сохраняется. Если модель
    ничего не скопировала, контент возвращается без изменений.
    """
    result_lines = {
        line.strip() for result in tool_results for line in result.splitlines() if line.strip()
    }
    first_lines = {
        result.strip().splitlines()[0].strip() for result in tool_results if result.strip()
    }
    kept: list[str] = []
    echoing = False
    dropped = False
    for line in content.splitlines():
        stripped = line.strip()
        if echoing and (not stripped or stripped in result_lines):
            continue
        echoing = stripped in first_lines
        if echoing:
            dropped = True
            continue
        kept.append(line)
    if not dropped:
        return content
    return "\n".join(kept).strip()


@dataclass(frozen=True)
class AgentRun:
    """Результат одного агентного цикла.

    ``exchange`` — все сообщения запуска: пользователь, пары «вызов
    инструмента → результат», финальный ответ модели. По нему приложение
    судит о качестве запуска; в чат-сессию хранилище берёт из него только
    реплики диалога (tool-обмен на диск не пишется).
    ``failed_tool_results`` — сколько вызовов инструментов завершились
    ошибкой: признак проблемного запуска для решения о повторе выше.
    """

    steps_used: int
    exchange: tuple[InferenceMessage, ...]
    final_answer: str | None
    stopped_by_limit: bool
    failed_tool_results: int = 0


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
        keep_steps: int = 3,
    ) -> None:
        self._inference = inference
        self._model = model
        self._system_prompt = system_prompt
        self._step_limit = step_limit
        self._keep_steps = keep_steps
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
            # Дата рендерится на каждый запуск: долгоживущий процесс не должен
            # рассказывать вчерашний день.
            InferenceMessage(
                role=MessageRole.SYSTEM,
                content=f"{self._system_prompt}\n\n{build_date_block()}",
            ),
            *history,
            user_message,
        ]
        exchange: list[InferenceMessage] = [user_message]
        empty_retries = 0
        failed_tool_results = 0
        # tool-сообщения каждого завершённого шага: индекс в messages плюс
        # вызов и статус — сырьё для компакции старших шагов.
        step_tools: list[list[tuple[int, ToolCall, bool]]] = []
        for step in range(1, self._step_limit + 1):
            self._compact_old_steps(messages, step_tools)
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
                # Дословное эхо результатов инструментов (свежих и из истории)
                # вырезается из финального ответа: пользователю и в сессию
                # уходит только собственный текст модели.
                answer = _strip_tool_echo(
                    response.content,
                    tuple(
                        message.content for message in messages if message.role is MessageRole.TOOL
                    ),
                )
                if answer != response.content:
                    self._logger.warning(
                        "agent_final_answer_tool_echo_stripped",
                        request_id=request_id,
                        step=step,
                    )
                if not answer.strip():
                    if empty_retries < _EMPTY_STEP_RETRIES:
                        empty_retries += 1
                        self._logger.warning(
                            "agent_step_empty_retry",
                            request_id=request_id,
                            step=step,
                            attempt=empty_retries,
                        )
                        continue
                    self._logger.warning(
                        "agent_step_empty_final",
                        request_id=request_id,
                        step=step,
                    )
                    raise EmptyInferenceResponseError(
                        "Inference provider returned an empty response"
                    )
                exchange.append(InferenceMessage(role=MessageRole.ASSISTANT, content=answer))
                self._log_run(request_id, started_at, steps=step, stopped=False)
                return AgentRun(
                    steps_used=step,
                    exchange=tuple(exchange),
                    final_answer=answer,
                    stopped_by_limit=False,
                    failed_tool_results=failed_tool_results,
                )
            messages.append(assistant)
            exchange.append(assistant)
            executed_results: list[ToolResult] = []
            for call in response.tool_calls:
                result = await self._execute(request_id, call, reporter)
                if not result.succeeded:
                    failed_tool_results += 1
                executed_results.append(result)
                result_message = InferenceMessage(
                    role=MessageRole.TOOL,
                    content=result.content,
                    tool_name=call.name,
                )
                messages.append(result_message)
                exchange.append(result_message)
            step_tools.append(
                [
                    (len(messages) - len(response.tool_calls) + i, call, result.succeeded)
                    for i, (call, result) in enumerate(
                        zip(response.tool_calls, executed_results, strict=True)
                    )
                ]
            )
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
            failed_tool_results=failed_tool_results,
        )

    def _compact_old_steps(
        self,
        messages: list[InferenceMessage],
        step_tools: list[list[tuple[int, ToolCall, bool]]],
    ) -> None:
        """Схлопнуть tool-выводы шагов, вышедших из окна последних K шагов.

        Коснулся только рабочего множества запуска (список ``messages``):
        записанный обмен и чат-сессия не меняются. Перезаписи идемпотентны:
        старый шаг даёт ту же сигнатуру при каждом пересчёте.
        """
        if self._keep_steps <= 0 or len(step_tools) <= self._keep_steps:
            return
        for entries in step_tools[: -self._keep_steps]:
            for index, call, succeeded in entries:
                message = messages[index]
                if message.role is not MessageRole.TOOL:
                    continue
                command = command_from_arguments(call.arguments)
                # Невалидные аргументи не тащат сырой JSON в сигнатуру.
                label = command if command is not None else str(call.name)
                status = "ok" if succeeded else "ошибка"
                messages[index] = replace(message, content=f"{label} → {status}")

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
