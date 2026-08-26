"""The ``ProcessUserMessage`` use case: агентная обработка сообщения в чат-сессии."""

import asyncio

import structlog

from bot.agent.loop import AgentLoop
from bot.agent.progress import AgentProgress
from bot.application.models import UserMessageRequest, UserMessageResponse
from bot.domain.ids import TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.sessions.store import ChatSessionStore
from bot.sessions.window import trim_to_window


class ApplicationService:
    """Processes user messages inside their chat session.

    Чат-сессия — единый длинный контекст: история каждого сообщения
    подмешивается к запросу, агентный цикл добирает шаги с инструментами.
    Обработка сообщений одного чата строго последовательна (блокировка на
    чат), разные чаты — параллельны.
    """

    def __init__(
        self,
        agent: AgentLoop,
        sessions: ChatSessionStore,
        history_limit: int,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._agent = agent
        self._sessions = sessions
        self._history_limit = history_limit
        self._logger = logger.bind(component="application_service")
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
        history = trim_to_window(
            self._sessions.load(request.chat_id),
            self._history_limit,
        )
        user_message = InferenceMessage(role=MessageRole.USER, content=request.text)
        run = await self._agent.run(request.request_id, history, user_message, progress)
        # Обмен сохраняется целиком: пользователь, пары «вызов → результат»,
        # финальный ответ. Незавершённые попытки (исключение) сессию не меняют.
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

    def _lock_for(self, chat_id: TelegramChatId) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock
