"""Отправка отформатированных текстов: MarkdownV2 в одном месте слоя."""

from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from bot.telegram.formatter import format_for_telegram

_PARSE_MODE = ParseMode.MARKDOWN_V2


async def send_formatted(message: Message, text: str) -> Message:
    """Отправить текст через единый форматтер; вернуть последнее сообщение.

    Длинный текст уходит несколькими частями (сплит + экранирование каждой,
    MarkdownV2); возврат — последнее отправленное сообщение, чтобы вызывающий
    мог продолжить работу с ним (например, править как статус-месседж).
    """
    parts = format_for_telegram(text)
    sent = await message.answer(parts[0], parse_mode=_PARSE_MODE)
    for part in parts[1:]:
        sent = await message.answer(part, parse_mode=_PARSE_MODE)
    return sent


async def edit_formatted(status: Message, text: str) -> Message:
    """Править статус-сообщение текстом; вернуть актуальное сообщение.

    Если правка невозможна (сообщение устарело или удалено) — фолбэк на новое
    сообщение. Текст длиннее одной части дописывается следующими сообщениями
    в обоих путях: ни одна часть не теряется.
    """
    parts = format_for_telegram(text)
    try:
        await status.edit_text(parts[0], parse_mode=_PARSE_MODE)
        current = status
    except TelegramAPIError:
        current = await status.answer(parts[0], parse_mode=_PARSE_MODE)
    for part in parts[1:]:
        current = await current.answer(part, parse_mode=_PARSE_MODE)
    return current
