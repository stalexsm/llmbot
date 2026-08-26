"""Typed application-layer DTOs."""

from dataclasses import dataclass

from bot.domain.ids import (
    RequestId,
    TelegramChatId,
    TelegramMessageId,
    TelegramUserId,
)


@dataclass(frozen=True)
class UserMessageRequest:
    """Input of the ``ProcessUserMessage`` use case."""

    request_id: RequestId
    user_id: TelegramUserId
    chat_id: TelegramChatId
    message_id: TelegramMessageId
    text: str


@dataclass(frozen=True)
class UserMessageResponse:
    """Output of the ``ProcessUserMessage`` use case."""

    request_id: RequestId
    text: str
    # Цикл остановлен по лимиту шагов: текста модели нет, пользователь получает
    # честное сообщение об остановке (текст принадлежит Telegram-слою).
    stopped_by_step_limit: bool = False
