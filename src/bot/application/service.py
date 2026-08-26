"""The ``ProcessUserMessage`` use case: агентная обработка сообщения в чат-сессии."""

import asyncio
import time

import structlog

from bot.application.errors import ApplicationError, EmptyInferenceResponseError
from bot.application.models import UserMessageRequest, UserMessageResponse
from bot.domain.ids import ModelId, RequestId, TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest
from bot.inference.provider import InferenceProvider
from bot.sessions.store import ChatSessionStore
from bot.sessions.window import trim_to_window


class ApplicationService:
    """Processes user messages inside their chat session.

    Чат-сессия — единый длинный контекст: история каждого сообщения
    подмешивается к запросу. Обработка сообщений одного чата строго
    последовательна (блокировка на чат), разные чаты — параллельны.
    """

    def __init__(
        self,
        inference: InferenceProvider,
        model: ModelId,
        sessions: ChatSessionStore,
        history_limit: int,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._inference = inference
        self._model = model
        self._sessions = sessions
        self._history_limit = history_limit
        self._logger = logger.bind(component="application_service")
        self._chat_locks: dict[TelegramChatId, asyncio.Lock] = {}

    async def process_message(self, request: UserMessageRequest) -> UserMessageResponse:
        async with self._lock_for(request.chat_id):
            return await self._process_locked(request)

    async def reset_session(self, chat_id: TelegramChatId) -> None:
        """Команда /new: обнулить чат-сессию, история отбрасывается."""
        async with self._lock_for(chat_id):
            self._sessions.reset(chat_id)
        self._logger.info("session_reset", chat_id=chat_id)

    async def _process_locked(self, request: UserMessageRequest) -> UserMessageResponse:
        history = trim_to_window(
            self._sessions.load(request.chat_id),
            self._history_limit,
        )
        user_message = InferenceMessage(role=MessageRole.USER, content=request.text)
        inference_request = InferenceRequest(
            request_id=request.request_id,
            model=self._model,
            messages=(*history, user_message),
        )
        started_at = time.monotonic()
        try:
            inference_response = await self._inference.generate(inference_request)
        except ApplicationError as exc:
            self._log_inference(
                request.request_id,
                started_at,
                status="error",
                error=type(exc).__name__,
            )
            raise
        if not inference_response.content.strip():
            self._log_inference(request.request_id, started_at, status="empty_response")
            raise EmptyInferenceResponseError("Inference provider returned an empty response")
        self._log_inference(request.request_id, started_at, status="success")
        # Обмен сохраняется целиком: незавершённые попытки сессию не меняют.
        assistant_message = InferenceMessage(
            role=MessageRole.ASSISTANT,
            content=inference_response.content,
        )
        self._sessions.append(request.chat_id, user_message, assistant_message)
        return UserMessageResponse(
            request_id=inference_response.request_id,
            text=inference_response.content,
        )

    def _lock_for(self, chat_id: TelegramChatId) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock

    def _log_inference(
        self,
        request_id: RequestId,
        started_at: float,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        event: dict[str, str | int] = {
            "request_id": request_id,
            "model": self._model,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "status": status,
        }
        if error is not None:
            event["error"] = error
        self._logger.info("inference_finished", **event)
