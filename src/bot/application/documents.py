"""Use cases над корпусом документов: индексация, список, удаление.

Владелец — ``TelegramUserId`` с границы Telegram-слоя (ADR-0002). Индексация
одного владельца сериализуется блокировкой (замена документа атомарна, поиск
во время индексации видит старый корпус); телом работы занимается RagService,
здесь — только очередь владельца и типизированные сценарии. Чат не
блокируется: вызов ``index`` выполняется фоновой задачей вызывающей стороны.
"""

import asyncio
from dataclasses import dataclass

import structlog

from bot.application.progress import DocumentIndexProgress
from bot.domain.ids import RequestId, TelegramUserId
from bot.rag.models import DocumentInfo, IndexedDocument
from bot.rag.service import RagService


@dataclass(frozen=True)
class DocumentUpload:
    """Вход сценария индексации: файл от владельца, ещё не Документ."""

    request_id: RequestId
    owner_id: TelegramUserId
    name: str
    content: bytes


@dataclass(frozen=True)
class DocumentIndexResult:
    """Итог индексации: имя документа и число его чанков."""

    name: str
    chunk_count: int

    @classmethod
    def from_indexed(cls, indexed: IndexedDocument) -> "DocumentIndexResult":
        return cls(indexed.document.name, indexed.chunk_count)


class DocumentService:
    """Индексация документов, корпус и удаление — по владельцу."""

    def __init__(self, rag: RagService, logger: structlog.stdlib.BoundLogger) -> None:
        self._rag = rag
        self._logger = logger.bind(component="document_service")
        self._owner_locks: dict[TelegramUserId, asyncio.Lock] = {}

    async def index(
        self,
        upload: DocumentUpload,
        progress: DocumentIndexProgress | None = None,
    ) -> DocumentIndexResult:
        """Проиндексировать файл владельца; загрузки одного владельца — по очереди.

        Прикладные ошибки (формат, лимиты, эмбеддинги, rag-БД) пробрасываются
        вызывающей стороне — та превращает их в понятные сообщения.
        """
        async with self._lock_for(upload.owner_id):
            indexed = await self._rag.index_document(
                upload.request_id,
                upload.owner_id,
                upload.name,
                upload.content,
                progress=progress,
            )
        self._logger.info(
            "document_indexed",
            request_id=upload.request_id,
            owner_id=int(upload.owner_id),
            chunks=indexed.chunk_count,
        )
        return DocumentIndexResult.from_indexed(indexed)

    def list_documents(self, owner_id: TelegramUserId) -> tuple[DocumentInfo, ...]:
        """Корпус владельца по алфавиту имён."""
        return self._rag.list_documents(owner_id)

    def delete_document(self, owner_id: TelegramUserId, name: str) -> None:
        """Удалить документ владельца вместе с чанками и эмбеддингами.

        Неизвестное имя — ``ApplicationError`` (``DocumentNotFoundError``).
        """
        self._rag.delete_document(owner_id, name)

    def clear_documents(self, owner_id: TelegramUserId) -> int:
        """Очистить корпус владельца; вернуть число удалённых документов."""
        return self._rag.clear_documents(owner_id)

    def _lock_for(self, owner_id: TelegramUserId) -> asyncio.Lock:
        lock = self._owner_locks.get(owner_id)
        if lock is None:
            lock = asyncio.Lock()
            self._owner_locks[owner_id] = lock
        return lock


__all__ = ["DocumentIndexResult", "DocumentService", "DocumentUpload"]
