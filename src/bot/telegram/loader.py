"""Загрузка файла документа из Telegram.

Telegram-детали (getFile, скачивание) живут здесь; хендлеры зависят только
от протокола ``DocumentLoader`` — в тестах он подменяется фейковкой.
"""

import io
from typing import Protocol

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.methods import GetFile


class DocumentLoader(Protocol):
    """Источник содержимого файла по ``file_id``."""

    async def load(self, bot: Bot, file_id: str) -> bytes: ...


class TelegramDocumentLoader:
    """Скачивает файл документа через Bot API в память."""

    async def load(self, bot: Bot, file_id: str) -> bytes:
        file = await bot.get_file(file_id)
        if file.file_path is None:
            raise TelegramAPIError(
                method=GetFile(file_id=file_id), message="Telegram returned no file path"
            )
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        return buffer.getvalue()
