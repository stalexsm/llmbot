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
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmbeddingError,
)
from bot.application.progress import (
    DocumentIndexProgress,
    IndexingStage,
    NullDocumentIndexProgress,
)
from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.inference.embeddings import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from bot.rag.chunking import chunk_pages
from bot.rag.extract import document_kind, extract_pages
from bot.rag.models import EMBEDDING_DIMENSION, DocumentInfo, IndexedDocument, SearchHit
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
        progress: DocumentIndexProgress | None = None,
    ) -> IndexedDocument:
        """Превратить файл в Документ: извлечь, нарезать, векторизовать, записать.

        Завершается документом, доступным поиску, либо прикладной ошибкой
        (неподдерживаемый формат, пустой или слишком большой документ).
        Повторная индексация того же имени заменяет документ атомарно.
        Наблюдатель стадий вызывается между шагами pipeline: извлечение,
        число чанков, эмбеддинги; его сбой индексацию не рвёт (гасится
        с записью в лог — статус пользователю доставит вызывающая сторона).
        """
        observer: DocumentIndexProgress = (
            progress if progress is not None else NullDocumentIndexProgress()
        )
        await self._report(observer, IndexingStage.EXTRACTING)
        if len(content) > self._max_file_bytes:
            raise DocumentTooLargeError("Document exceeds the file size limit")
        pages = extract_pages(name, content)
        total_chars = sum(len(page.text) for page in pages)
        if total_chars > self._max_text_chars:
            raise DocumentTooLargeError("Document exceeds the extracted text limit")

        chunks = chunk_pages(
            pages,
            target_chars=self._chunk_target_chars,
            overlap_chars=self._chunk_overlap_chars,
        )
        if len(chunks) > self._max_chunks:
            raise DocumentTooLargeError("Document exceeds the chunk count limit")
        texts = [chunk.text for chunk in chunks]
        await self._report(observer, IndexingStage.CHUNKED, chunks=len(chunks))

        await self._report(observer, IndexingStage.EMBEDDING, chunks=len(chunks))
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

    def list_documents(self, owner_id: TelegramUserId) -> tuple[DocumentInfo, ...]:
        """Корпус владельца: проиндексированные документы по алфавиту."""
        return self._store.list_documents(owner_id)

    def delete_document(self, owner_id: TelegramUserId, name: str) -> None:
        """Удалить документ владельца вместе с чанками и эмбеддингами.

        Неизвестное имя — ``DocumentNotFoundError``: вызывающая сторона
        отвечает пользователю дружелюбной ошибкой со списком корпуса.
        """
        if not self._store.delete_document(owner_id, name):
            raise DocumentNotFoundError(f"Document not found: {name}")

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

    @staticmethod
    async def _report(
        observer: DocumentIndexProgress,
        stage: IndexingStage,
        *,
        chunks: int = 0,
    ) -> None:
        """Сообщить стадию наблюдателю; его сбой — не повод рвать индексацию."""
        try:
            await observer.on_stage(stage, chunks=chunks)
        except Exception:
            # Правка статус-месседжа не удалась (Telegram недоступен) —
            # на результат индексации это влиять не должно.
            return

    def _validate_embeddings(self, response: EmbeddingResponse, *, expected: int) -> None:
        """Ответ эмбеддинг-модели обязан совпадать по числу и размерности."""
        if len(response.embeddings) != expected:
            raise EmbeddingError("Embedding provider returned a wrong number of vectors")
        for vector in response.embeddings:
            if len(vector) != EMBEDDING_DIMENSION:
                raise EmbeddingError(f"Embedding dimension must be {EMBEDDING_DIMENSION}")
