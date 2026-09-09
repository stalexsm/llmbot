"""Прогресс индексации: стадии pipeline без знания пользователя стадий.

Аналог ``AgentProgress`` в agent-слое: RagService отчитывается стадиями
(извлечение → чанки → эмбеддинги), потребитель — Telegram-слой — рисует их
как угодно; по умолчанию это заглушка ``NullDocumentIndexProgress``. Тексты
стадий принадлежат слоям выше: сюда доходит только типизированная стадия.
Контракт живёт в application-слое, чтобы rag-сервис и telegram-реализация
встретились в композиционном корне без прямых импортов rag из telegram.
"""

from enum import StrEnum
from typing import Protocol


class IndexingStage(StrEnum):
    """Стадии индексации в порядке исполнения pipeline."""

    EXTRACTING = "extracting"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"


class DocumentIndexProgress(Protocol):
    """Наблюдатель стадий индексации одного документа."""

    async def on_stage(self, stage: IndexingStage, *, chunks: int = 0) -> None: ...


class NullDocumentIndexProgress:
    """Заглушка: индексация без наблюдателя (сценарии вне Telegram)."""

    async def on_stage(self, stage: IndexingStage, *, chunks: int = 0) -> None:
        return None
