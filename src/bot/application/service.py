"""The ``ProcessUserMessage`` use case: агентная обработка сообщения в чат-сессии."""

import asyncio

import structlog

from bot.agent.command_class import command_from_arguments
from bot.agent.loop import AgentLoop, AgentRun
from bot.agent.progress import AgentProgress
from bot.application.models import UserMessageRequest, UserMessageResponse
from bot.domain.ids import TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ExecutionContext
from bot.metrics.collector import RunMetrics
from bot.sessions.store import ChatSessionStore
from bot.sessions.window import trim_to_window


def _dialogue_turns(
    history: tuple[InferenceMessage, ...],
    current_text: str,
) -> tuple[str, ...]:
    """Реплики диалога для контекста выполнения: история и текущий вопрос.

    Формат — «Роль: текст»; из истории берутся только реплики диалога
    (user/assistant, tool-обмен в сессиях и не хранится). Последняя реплика —
    текущий вопрос пользователя: без него переписывание не разрешит
    местоимения именно этого вопроса.
    """
    labels = {MessageRole.USER: "Пользователь", MessageRole.ASSISTANT: "Ассистент"}
    turns = [
        f"{labels[message.role]}: {message.content}"
        for message in history
        if message.role in labels
    ]
    turns.append(f"Пользователь: {current_text}")
    return tuple(turns)


class ApplicationService:
    """Processes user messages inside their chat session.

    Чат-сессия — единый длинный контекст: история каждого сообщения
    подмешивается к запросу, агентный цикл добирает шаги с инструментами.
    Обработка сообщений одного чата строго последовательна (блокировка на
    чат), разные чаты — параллельны.
    """

    # Запуск, где модель «сдалась» после провалившихся команд, не пишется
    # в сессию и повторяется один раз: неудачный обмен в истории учит
    # крошечные модели повторять отказ вместо работы (компиляция брака).
    _GIVEUP_MARKERS = (
        "я не знаю",
        "не удалось",
        "не доступна",
        "недоступна",
        "недоступен",
        "недоступны",
        "не смог",
        "не могу получит",
    )
    _GIVEUP_RETRIES = 1

    def __init__(
        self,
        agent: AgentLoop,
        sessions: ChatSessionStore,
        history_limit: int,
        logger: structlog.stdlib.BoundLogger,
        metrics: RunMetrics,
    ) -> None:
        self._agent = agent
        self._sessions = sessions
        self._history_limit = history_limit
        self._logger = logger.bind(component="application_service")
        self._metrics = metrics
        self._chat_locks: dict[TelegramChatId, asyncio.Lock] = {}

    async def process_message(
        self,
        request: UserMessageRequest,
        progress: AgentProgress | None = None,
    ) -> UserMessageResponse:
        async with self._lock_for(request.chat_id):
            return await self._process_locked(request, progress)

    async def reset_session(self, chat_id: TelegramChatId) -> None:
        """Команда /new: обнулить чат-сессию, история отбрасывается."""
        async with self._lock_for(chat_id):
            self._sessions.reset(chat_id)
        self._logger.info("session_reset", chat_id=chat_id)

    async def _process_locked(
        self,
        request: UserMessageRequest,
        progress: AgentProgress | None,
    ) -> UserMessageResponse:
        # Запуск закрывается метрикой при любом исходе: штатный ответ, лимит
        # шагов или исключение — всегда одна запись run на request_id.
        run: AgentRun | None = None
        try:
            history = trim_to_window(
                self._sessions.load(request.chat_id),
                self._history_limit,
            )
            user_message = InferenceMessage(role=MessageRole.USER, content=request.text)
            # Скоуп владельца (ADR-0002) доходит до инструментов контекстом
            # выполнения: search_documents ищет только в его корпусе; реплики
            # диалога нужны переписыванию поискового запроса.
            context = ExecutionContext(
                owner_id=request.user_id,
                recent_turns=_dialogue_turns(history, request.text),
            )
            run = await self._agent.run(
                request.request_id, history, user_message, progress, context
            )
            # Отказ-подобный ответ — не пишем его в сессию (иначе он учит модель
            # сдаваться) и пробуем ещё раз: и когда команды провалились, и когда
            # модель даже не попыталась ими воспользоваться.
            for attempt in range(1, self._GIVEUP_RETRIES + 2):
                last_chance = attempt > self._GIVEUP_RETRIES
                if last_chance or not self._is_broken_run(run):
                    break
                self._logger.warning(
                    "agent_run_giveup_retry",
                    request_id=request.request_id,
                    attempt=attempt,
                )
                run = await self._agent.run(
                    request.request_id, history, user_message, progress, context
                )
            # В сессию попадает только диалог из обмена — сообщение пользователя
            # и финальный ответ; tool-обмен хранилище отфильтровывает само.
            # Незавершённые попытки (исключение) сессию не меняют.
            if not self._is_broken_run(run):
                self._sessions.append(request.chat_id, *run.exchange)
            if run.stopped_by_limit:
                self._logger.warning(
                    "agent_stopped_by_step_limit",
                    request_id=request.request_id,
                    steps=run.steps_used,
                )
                return UserMessageResponse(
                    request_id=request.request_id,
                    text="",
                    stopped_by_step_limit=True,
                )
            self._logger.info(
                "message_processed",
                request_id=request.request_id,
                steps=run.steps_used,
            )
            assert run.final_answer is not None
            return UserMessageResponse(
                request_id=request.request_id,
                text=run.final_answer,
            )
        finally:
            self._metrics.finish_run(
                request.request_id,
                success=run is not None and not run.stopped_by_limit,
            )

    @classmethod
    def _is_broken_run(cls, run: AgentRun) -> bool:
        """Похоже ли, что запуск «сдался», а не отработал задачу.

        Брак — отказ-подобный финал, за которым либо провалившиеся команды,
        либо полное отсутствие попыток, либо только чтение скиллов (прочитал
        инструкцию и бросил, не дойдя до данных). Честное «не знаю» после
        реальной работы с данными браком не считается.
        """
        if run.final_answer is None or not cls._is_giveup(run.final_answer):
            return False
        tools_used = sum(1 for message in run.exchange if message.role is MessageRole.TOOL)
        if run.failed_tool_results > 0 or tools_used == 0:
            return True
        return cls._only_skill_reads(run.exchange)

    @staticmethod
    def _only_skill_reads(exchange: tuple[InferenceMessage, ...]) -> bool:
        """Все ли вызовы инструментов в обмене — чтение файлов скиллов."""
        calls = [
            call
            for message in exchange
            if message.role is MessageRole.ASSISTANT
            for call in message.tool_calls
        ]
        if not calls:
            return False
        for call in calls:
            command = command_from_arguments(call.arguments)
            if command is None or not command.startswith("cat skills/"):
                return False
        return True

    @classmethod
    def _is_giveup(cls, text: str) -> bool:
        """Похоже ли сообщение на отказ-после-неудач (в нижнем регистре)."""
        lowered = text.lower()
        return any(marker in lowered for marker in cls._GIVEUP_MARKERS)

    def _lock_for(self, chat_id: TelegramChatId) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock
