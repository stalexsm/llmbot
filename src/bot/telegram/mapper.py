"""Mapping between aiogram objects and application DTOs."""

from uuid import uuid4

from aiogram.types import Message

from bot.application.models import UserMessageRequest
from bot.domain.ids import (
    RequestId,
    TelegramChatId,
    TelegramMessageId,
    TelegramUserId,
)


def to_user_request(message: Message) -> UserMessageRequest:
    """Convert a Telegram text message into an application-level request."""
    from_user_id = message.from_user.id if message.from_user is not None else message.chat.id
    return UserMessageRequest(
        request_id=RequestId(str(uuid4())),
        user_id=TelegramUserId(from_user_id),
        chat_id=TelegramChatId(message.chat.id),
        message_id=TelegramMessageId(message.message_id),
        text=message.text or "",
    )
