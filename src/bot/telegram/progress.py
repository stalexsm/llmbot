"""Статус-месседж индексации: один редактируемый текст со троттлингом.

Стадии приходят от RagService через контракт ``DocumentIndexProgress``;
здесь они превращаются в понятные тексты и правкой одного сообщения
показываются пользователю: правки не чаще ``min_interval_seconds``,
невозможность правки (старое сообщение недоступно) — фолбэк на новое
сообщение. Финальный итог (готово или ошибка) показывается всегда.
"""

import time

import structlog
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from bot.application.progress import IndexingStage


class TelegramIndexProgress:
    """Реализация ``DocumentIndexProgress`` поверх сообщения Telegram."""

    def __init__(
        self,
        status: Message,
        document_name: str,
        logger: structlog.stdlib.BoundLogger,
        *,
        min_interval_seconds: float = 1.5,
    ) -> None:
        self._status = status
        self._name = document_name
        self._logger = logger.bind(component="telegram_index_progress")
        self._min_interval = min_interval_seconds
        self._last_edit = 0.0

    async def on_stage(self, stage: IndexingStage, *, chunks: int = 0) -> None:
        await self._update(self._stage_text(stage, chunks), force=False)

    async def finish_success(self, chunk_count: int) -> None:
        await self._update(
            f"✅ Документ готов: {self._name} — {chunk_count} чанков.\n"
            "Теперь по нему можно спрашивать.",
            force=True,
        )

    async def finish_error(self, text: str) -> None:
        await self._update(text, force=True)

    def _stage_text(self, stage: IndexingStage, chunks: int) -> str:
        if stage is IndexingStage.EXTRACTING:
            return f"📄 {self._name} — извлекаю текст…"
        if stage is IndexingStage.CHUNKED:
            return f"📄 {self._name} — текст разбит на {chunks} чанков."
        return f"📄 {self._name} — считаю эмбеддинги ({chunks} чанков)…"

    async def _update(self, text: str, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_edit < self._min_interval:
            return
        try:
            await self._status.edit_text(text)
        except TelegramAPIError:
            # Правка невозможна (сообщение устарело/удалено) — новое сообщение.
            self._logger.info("index_status_edit_failed", status="fallback_new_message")
            self._status = await self._status.answer(text)
        self._last_edit = time.monotonic()
