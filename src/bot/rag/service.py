"""RAG-сервис: полный pipeline «извлечение → чанкинг → эмбеддинги → БД → поиск».

Верх RAG-слоя: потребители (application-сценарии и инструмент агента) видят
только его. Индексация превращает файл в Документ или завершается понятной
ошибкой; поиск возвращает чанки корпуса владельца с Источниками либо пустой
результат ниже порога близости. Эмбеддинги — за портом ``EmbeddingProvider``:
в тестах подменяется детерминированной фейковкой, в рантайме это адаптер
Ollama из композиционного корня.
"""

import structlog

from bot.application.errors import (
    DocumentTooLargeError,
    EmbeddingError,
)
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.inference.embeddings import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from bot.rag.chunking import chunk_text
from bot.rag.extract import document_kind, extract_text
from bot.rag.models import EMBEDDING_DIMENSION, Chunk, IndexedDocument, SearchHit
from bot.rag.store import RagStore


class RagService:
    """Индексация документов и поиск по корпусу одного владельца."""

    def __init__(
        self,
        store: RagStore,
        embeddings: EmbeddingProvider,
        *,
        model: ModelId,
        chunk_target_chars: int,
        chunk_overlap_chars: int,
        top_k: int,
        overfetch: int,
        min_similarity: float,
        max_file_bytes: int,
        max_text_chars: int,
        max_chunks: int,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._model = model
        self._chunk_target_chars = chunk_target_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._top_k = top_k
        self._overfetch = overfetch
        self._min_similarity = min_similarity
        self._max_file_bytes = max_file_bytes
        self._max_text_chars = max_text_chars
        self._max_chunks = max_chunks
        self._logger = logger.bind(component="rag_service")

    async def index_document(
        self,
        request_id: RequestId,
        owner_id: TelegramUserId,
        name: str,
        content: bytes,
    ) -> IndexedDocument:
        """Превратить файл в Документ: извлечь, нарезать, векторизовать, записать.

        Завершается документом, доступным поиску, либо прикладной ошибкой
        (неподдерживаемый формат, пустой или слишком большой документ).
        Повторная индексация того же имени заменяет документ атомарно.
        """
        if len(content) > self._max_file_bytes:
            raise DocumentTooLargeError("Document exceeds the file size limit")
        text = extract_text(name, content)
        if len(text) > self._max_text_chars:
            raise DocumentTooLargeError("Document exceeds the extracted text limit")

        texts = chunk_text(
            text,
            target_chars=self._chunk_target_chars,
            overlap_chars=self._chunk_overlap_chars,
        )
        if len(texts) > self._max_chunks:
            raise DocumentTooLargeError("Document exceeds the chunk count limit")
        chunks = [
            Chunk(position=position, text=chunk_text_value)
            for position, chunk_text_value in enumerate(texts)
        ]

        response = await self._embeddings.embed(
            EmbeddingRequest(
                request_id=request_id,
                model=self._model,
                texts=tuple(texts),
            )
        )
        self._validate_embeddings(response, expected=len(chunks))

        document = self._store.replace_document(
            owner_id,
            name,
            document_kind(name),
            chunks,
            response.embeddings,
        )
        self._logger.info(
            "rag_document_indexed",
            request_id=request_id,
            owner_id=int(owner_id),
            document_id=document.id,
            chunks=len(chunks),
        )
        return IndexedDocument(document=document, chunk_count=len(chunks))

    async def search(
        self,
        request_id: RequestId,
        owner_id: TelegramUserId,
        query: str,
    ) -> tuple[SearchHit, ...]:
        """Найти top-K чанков корпуса владельца по Поисковому запросу.

        Ниже порога косинусной близости возвращается пустой кортеж —
        явное «ничего не найдено» для вызывающей стороны.
        """
        if not query.strip():
            return ()
        response = await self._embeddings.embed(
            EmbeddingRequest(request_id=request_id, model=self._model, texts=(query,))
        )
        self._validate_embeddings(response, expected=1)
        hits = self._store.search(
            owner_id,
            response.embeddings[0],
            top_k=self._top_k,
            overfetch=self._overfetch,
            min_similarity=self._min_similarity,
        )
        self._logger.info(
            "rag_search",
            request_id=request_id,
            owner_id=int(owner_id),
            hits=len(hits),
        )
        return hits

    def _validate_embeddings(self, response: EmbeddingResponse, *, expected: int) -> None:
        """Ответ эмбеддинг-модели обязан совпадать по числу и размерности."""
        if len(response.embeddings) != expected:
            raise EmbeddingError("Embedding provider returned a wrong number of vectors")
        for vector in response.embeddings:
            if len(vector) != EMBEDDING_DIMENSION:
                raise EmbeddingError(f"Embedding dimension must be {EMBEDDING_DIMENSION}")
