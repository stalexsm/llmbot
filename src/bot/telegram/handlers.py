"""Telegram adapter: aiogram handlers around the application service."""

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.application.errors import ApplicationError
from bot.application.service import ApplicationService
from bot.domain.ids import TelegramChatId
from bot.telegram.mapper import to_user_request
from bot.telegram.progress import TelegramCommandProgress
from bot.telegram.splitter import split_long_text

_START_TEXT = (
    "Привет! Я автономный агент на локальной языковой модели (Ollama).\n"
    "Веду диалог с памятью: у каждого чата своя чат-сессия, история\n"
    "сохраняется между сообщениями и перезапусками бота.\n"
    "Чтобы решить задачу, сам выполняю консольные команды\n"
    "(инструмент exec) и при необходимости пользуюсь скиллами —\n"
    "файлами с готовыми инструкциями.\n"
    "Команда /new начинает новую чат-сессию: история сбрасывается."
)

_NEW_CHAT_TEXT = "🆕 Новый чат: история диалога сброшена."

_ERROR_TEXT = "Не удалось получить ответ модели. Попробуйте повторить запрос позже."

_STEP_LIMIT_TEXT = (
    "⏹ Достигнут лимит шагов агента: задача остановлена.\n"
    "Уточните запрос или начните новый чат командой /new."
)


class TelegramHandlers:
    """Routes Telegram updates to the application layer and back."""

    def __init__(
        self,
        service: ApplicationService,
        logger: structlog.stdlib.BoundLogger,
        allowed_chat_ids: frozenset[TelegramChatId],
    ) -> None:
        self._service = service
        self._logger = logger.bind(component="telegram_handlers")
        # Пустой список — бот отвечает всем; заполненный — только перечисленным чатам.
        # Политика живёт в Settings; хендлер получает уже решённое множество.
        self._allowed_chat_ids = allowed_chat_ids

    def register(self, router: Router) -> None:
        router.message.register(self.handle_start, CommandStart())
        router.message.register(self.handle_new, Command("new"))
        router.message.register(self.handle_text, F.text, ~F.text.startswith("/"))

    def _chat_not_allowed(self, message: Message) -> bool:
        """Чат не входит в allowlist; чужой чат логируется и молча игнорируется."""
        if not self._allowed_chat_ids:
            return False
        if TelegramChatId(message.chat.id) in self._allowed_chat_ids:
            return False
        self._logger.info("chat_not_allowed", chat_id=message.chat.id)
        return True

    async def handle_start(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        self._logger.info("start_command_received", chat_id=message.chat.id)
        await message.answer(_START_TEXT)

    async def handle_new(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        try:
            await self._service.reset_session(TelegramChatId(message.chat.id))
        except ApplicationError:
            self._logger.error(
                "session_reset_failed",
                chat_id=message.chat.id,
                status="error",
            )
            await message.answer(_ERROR_TEXT)
            return
        self._logger.info("new_command_received", chat_id=message.chat.id)
        await message.answer(_NEW_CHAT_TEXT)

    async def handle_text(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        request = to_user_request(message)
        self._logger.info("message_received", request_id=request.request_id)
        progress = TelegramCommandProgress(source=message, logger=self._logger)
        try:
            response = await self._service.process_message(request, progress)
        except ApplicationError as exc:
            self._logger.error(
                "reply_failed",
                request_id=request.request_id,
                error=type(exc).__name__,
                status="error",
            )
            await message.answer(_ERROR_TEXT)
            return
        text = _STEP_LIMIT_TEXT if response.stopped_by_step_limit else response.text
        for part in split_long_text(text):
            await message.answer(part)
        self._logger.info("reply_sent", request_id=response.request_id, status="success")
