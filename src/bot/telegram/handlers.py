"""Telegram adapter: aiogram handlers around the application service."""

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.application.errors import ApplicationError
from bot.application.service import ApplicationService
from bot.domain.ids import TelegramChatId
from bot.telegram.mapper import to_user_request

_START_TEXT = (
    "Привет! Я подключён к локальной языковой модели через Ollama.\n"
    "Отправьте текст — я отвечу с учётом контекста нашего диалога.\n"
    "Команда /new начнёт новый чат без истории."
)

_NEW_CHAT_TEXT = "🆕 Новый чат: история диалога сброшена."

_ERROR_TEXT = "Не удалось получить ответ модели. Попробуйте повторить запрос позже."


class TelegramHandlers:
    """Routes Telegram updates to the application layer and back."""

    def __init__(
        self,
        service: ApplicationService,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._service = service
        self._logger = logger.bind(component="telegram_handlers")

    def register(self, router: Router) -> None:
        router.message.register(self.handle_start, CommandStart())
        router.message.register(self.handle_new, Command("new"))
        router.message.register(self.handle_text, F.text, ~F.text.startswith("/"))

    async def handle_start(self, message: Message) -> None:
        self._logger.info("start_command_received", chat_id=message.chat.id)
        await message.answer(_START_TEXT)

    async def handle_new(self, message: Message) -> None:
        await self._service.reset_session(TelegramChatId(message.chat.id))
        self._logger.info("new_command_received", chat_id=message.chat.id)
        await message.answer(_NEW_CHAT_TEXT)

    async def handle_text(self, message: Message) -> None:
        request = to_user_request(message)
        self._logger.info("message_received", request_id=request.request_id)
        try:
            response = await self._service.process_message(request)
        except ApplicationError as exc:
            self._logger.error(
                "reply_failed",
                request_id=request.request_id,
                error=type(exc).__name__,
                status="error",
            )
            await message.answer(_ERROR_TEXT)
            return
        await message.answer(response.text)
        self._logger.info("reply_sent", request_id=response.request_id, status="success")
