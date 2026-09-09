"""Доменные модели RAG-слоя: чанки, документы, результаты поиска.

Терминология — по глоссарию ``CONTEXT.md``: Документ, Владелец, Чанк,
Эмбеддинг, Индексация, Источник.
"""

from dataclasses import dataclass
from enum import StrEnum

from bot.domain.ids import DocumentId, TelegramUserId

# Размерность эмбеддингов модели bge-m3; она же — размер вектора в схеме
# vec0 (см. ``bot.rag.schema`` и миграции). Смена модели эмбеддингов с другой
# размерностью потребует новой ревизии миграции.
EMBEDDING_DIMENSION = 1024


class DocumentKind(StrEnum):
    """Тип документа: расширение без точки; расширяется парсерами тикета 06."""

    TXT = "txt"
    MD = "md"


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа: единица эмбеддинга и единица результата поиска.

    ``position`` — порядковый номер внутри документа; ``page`` — номер
    страницы, когда источник её знает (PDF), для .txt всегда ``None``.
    """

    position: int
    text: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentInfo:
    """Проиндексированный Документ: владелец, имя, тип, момент создания."""

    id: DocumentId
    owner_id: TelegramUserId
    name: str
    kind: DocumentKind
    created_at: str


@dataclass(frozen=True)
class IndexedDocument:
    """Результат индексации: документ и число его чанков."""

    document: DocumentInfo
    chunk_count: int


@dataclass(frozen=True)
class SearchHit:
    """Один результат поиска: чанк вместе с Источником и близостью.

    ``similarity`` — косинусная близость запроса и чанка в [0, 1].
    """

    document_name: str
    position: int
    text: str
    similarity: float
    page: int | None = None
