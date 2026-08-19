"""Unit tests for the Telegram→application mapper."""

from aiogram.types import Message

from bot.domain.ids import (
    TelegramChatId,
    TelegramMessageId,
    TelegramUserId,
)
from bot.telegram.mapper import to_user_request
from tests.fakes import make_telegram_message


def test_maps_telegram_fields() -> None:
    message = make_telegram_message("Привет", message_id=42, chat_id=100)

    request = to_user_request(message)

    assert request.user_id == TelegramUserId(7)
    assert request.chat_id == TelegramChatId(100)
    assert request.message_id == TelegramMessageId(42)
    assert request.text == "Привет"


def test_request_ids_are_unique() -> None:
    first = to_user_request(make_telegram_message("Первое"))
    second = to_user_request(make_telegram_message("Второе"))

    assert first.request_id != second.request_id


def test_message_without_from_user_falls_back_to_chat_id() -> None:
    message = Message.model_validate(
        {
            "message_id": 43,
            "date": 1735689600,
            "chat": {"id": 200, "type": "private"},
            "text": "Привет",
        }
    )

    request = to_user_request(message)

    assert request.user_id == TelegramUserId(200)
