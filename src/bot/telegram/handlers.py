"""Telegram adapter: aiogram handlers around the application service."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

import structlog
from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from bot.application.documents import DocumentService, DocumentUpload
from bot.application.errors import (
    ApplicationError,
    CorruptedDocumentError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    RagError,
    UnsupportedDocumentError,
)
from bot.application.service import ApplicationService
from bot.domain.ids import RequestId, TelegramChatId, TelegramUserId
from bot.rag.models import DocumentInfo
from bot.telegram.loader import DocumentLoader
from bot.telegram.mapper import to_user_request
from bot.telegram.progress import TelegramIndexProgress
from bot.telegram.splitter import split_long_text

_START_TEXT = (
    "Привет! Я автономный агент на локальной языковой модели (Ollama).\n"
    "Веду диалог с памятью: у каждого чата своя чат-сессия, история\n"
    "сохраняется между сообщениями и перезапусками бота.\n"
    "Чтобы решить задачу, сам выполняю консольные команды\n"
    "(инструмент execute_command) и при необходимости пользуюсь скиллами —\n"
    "файлами с готовыми инструкциями.\n"
    "Команда /new начинает новую чат-сессию: история сбрасывается."
)

_NEW_CHAT_TEXT = "🆕 Новый чат: история диалога сброшена."

_ERROR_TEXT = "Не удалось получить ответ модели. Попробуйте повторить запрос позже."

_STEP_LIMIT_TEXT = (
    "⏹ Достигнут лимит шагов агента: задача остановлена.\n"
    "Уточните запрос или начните новый чат командой /new."
)

_DOCUMENT_RECEIVED_TEXT = "📄 Документ получен: {name}. Индексирую…"
_DOCUMENT_DOWNLOAD_ERROR_TEXT = (
    "⚠ Не удалось скачать файл из Telegram. Попробуйте отправить документ ещё раз."
)
_NO_AUTHOR_TEXT = (
    "⚠ Не удалось определить автора сообщения. "
    "Документы привязаны к пользователю, поэтому без автора они не принимаются."
)
_NO_DOCUMENTS_TEXT = (
    "У вас пока нет документов. Пришлите файл .txt, .md, .pdf или .docx — я его проиндексирую."
)
_DOCUMENTS_HEADER_TEXT = "📚 Ваши документы:"
_DELETE_USAGE_TEXT = "Использование: /delete имя-файла — удалить документ из вашего корпуса."
_DOCUMENT_DELETED_TEXT = "🗑 Документ удалён: {name}."
_DOCUMENT_NOT_FOUND_TEXT = "Документ «{name}» не найден."


class TelegramHandlers:
    """Routes Telegram updates to the application layer and back."""

    def __init__(
        self,
        service: ApplicationService,
        documents: DocumentService,
        document_loader: DocumentLoader,
        logger: structlog.stdlib.BoundLogger,
        allowed_chat_ids: frozenset[TelegramChatId],
    ) -> None:
        self._service = service
        self._documents = documents
        self._document_loader = document_loader
        self._logger = logger.bind(component="telegram_handlers")
        # Пустой список — бот отвечает всем; заполненный — только перечисленным чатам.
        # Политика живёт в Settings; хендлер получает уже решённое множество.
        self._allowed_chat_ids = allowed_chat_ids
        # Держим ссылки на фоновые задачи индексации, пока они живы: голый
        # create_task сборщик мусора может прибить до завершения.
        self._indexing_tasks: set[asyncio.Task[None]] = set()

    def register(self, router: Router) -> None:
        router.message.register(self.handle_start, CommandStart())
        router.message.register(self.handle_new, Command("new"))
        router.message.register(self.handle_documents, Command("documents"))
        router.message.register(self.handle_delete, Command("delete"))
        router.message.register(self.handle_document, F.document)
        router.message.register(self.handle_text, F.text, ~F.text.startswith("/"))

    def _chat_not_allowed(self, message: Message) -> bool:
        """Чат не входит в allowlist; чужой чат логируется и молча игнорируется."""
        if not self._allowed_chat_ids:
            return False
        if TelegramChatId(message.chat.id) in self._allowed_chat_ids:
            return False
        self._logger.info("chat_not_allowed", chat_id=message.chat.id)
        return True

    async def handle_start(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        self._logger.info("start_command_received", chat_id=message.chat.id)
        await message.answer(_START_TEXT)

    async def handle_new(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        try:
            await self._service.reset_session(TelegramChatId(message.chat.id))
        except ApplicationError:
            self._logger.error(
                "session_reset_failed",
                chat_id=message.chat.id,
                status="error",
            )
            await message.answer(_ERROR_TEXT)
            return
        self._logger.info("new_command_received", chat_id=message.chat.id)
        await message.answer(_NEW_CHAT_TEXT)

    async def handle_text(self, message: Message) -> None:
        if self._chat_not_allowed(message):
            return
        request = to_user_request(message)
        self._logger.info("message_received", request_id=request.request_id)
        try:
            # Прогресс шагов в чат не выводится: выполняемые команды —
            # внутренняя кухня агента; цикл использует NullProgress.
            response = await self._service.process_message(request)
        except ApplicationError as exc:
            self._logger.error(
                "reply_failed",
                request_id=request.request_id,
                error=type(exc).__name__,
                status="error",
            )
            await message.answer(_ERROR_TEXT)
            return
        text = _STEP_LIMIT_TEXT if response.stopped_by_step_limit else response.text
        for part in split_long_text(text):
            await message.answer(part)
        self._logger.info("reply_sent", request_id=response.request_id, status="success")

    async def handle_document(self, message: Message) -> None:
        """Документ от пользователя: мгновенный ответ, индексация — фоном."""
        if self._chat_not_allowed(message):
            return
        document = message.document
        if document is None:  # фильтр F.document гарантирует документ; страховка для ty
            return
        name = document.file_name if document.file_name else "документ"
        if message.from_user is None:
            # Владелец документа — TelegramUserId с границы Telegram-слоя;
            # сообщение без автора не принимается (ADR-0002).
            self._logger.info("document_without_author", chat_id=message.chat.id)
            await message.answer(_NO_AUTHOR_TEXT)
            return
        owner_id = TelegramUserId(message.from_user.id)
        request_id = RequestId(str(uuid4()))
        self._logger.info(
            "document_received",
            request_id=request_id,
            owner_id=int(owner_id),
            chat_id=message.chat.id,
        )
        # Мгновенный ответ: индексация идёт фоновой задачей, чат не блокируется.
        status = await message.answer(_DOCUMENT_RECEIVED_TEXT.format(name=name))
        task = asyncio.create_task(
            self._index_in_background(message, status, owner_id, name, document.file_id, request_id)
        )
        self._indexing_tasks.add(task)
        task.add_done_callback(self._indexing_tasks.discard)

    async def _index_in_background(
        self,
        message: Message,
        status: Message,
        owner_id: TelegramUserId,
        name: str,
        file_id: str,
        request_id: RequestId,
    ) -> None:
        """Фоновая индексация одного документа; итог — тем же статус-месседжем."""
        progress = TelegramIndexProgress(status, name, self._logger)
        bot = message.bot
        try:
            if bot is None:
                raise RuntimeError("document update has no bot instance")
            content = await self._document_loader.load(bot, file_id)
        except (TelegramAPIError, OSError, RuntimeError) as exc:
            self._logger.error(
                "document_download_failed",
                request_id=request_id,
                error=type(exc).__name__,
                status="error",
            )
            await self._finish_safely(
                lambda: progress.finish_error(_DOCUMENT_DOWNLOAD_ERROR_TEXT), request_id=request_id
            )
            return
        try:
            result = await self._documents.index(
                DocumentUpload(
                    request_id=request_id, owner_id=owner_id, name=name, content=content
                ),
                progress,
            )
        except ApplicationError as exc:
            error_text = _indexing_error_text(exc)
            self._logger.error(
                "document_index_failed",
                request_id=request_id,
                error=type(exc).__name__,
                status="error",
            )
            await self._finish_safely(
                lambda: progress.finish_error(error_text), request_id=request_id
            )
            return
        self._logger.info(
            "document_indexed",
            request_id=request_id,
            owner_id=int(owner_id),
            status="success",
        )
        await self._finish_safely(
            lambda: progress.finish_success(result.chunk_count), request_id=request_id
        )

    async def _finish_safely(
        self,
        call: Callable[[], Awaitable[None]],
        *,
        request_id: RequestId,
    ) -> None:
        """Доставить итог в статус-месседж; сбой Telegram не рвёт фоновую задачу."""
        try:
            await call()
        except (TelegramAPIError, OSError) as exc:
            # Статус уже не достать ни правкой, ни новым сообщением — остаётся лог.
            self._logger.error(
                "document_status_delivery_failed",
                request_id=request_id,
                error=type(exc).__name__,
                status="error",
            )

    async def handle_documents(self, message: Message) -> None:
        """Команда /documents: корпус документов владельца."""
        if self._chat_not_allowed(message):
            return
        owner_id = await self._owner_or_answer(message)
        if owner_id is None:
            return
        try:
            documents = self._documents.list_documents(owner_id)
        except ApplicationError:
            self._logger.error("documents_list_failed", chat_id=message.chat.id, status="error")
            await message.answer(_ERROR_TEXT)
            return
        if not documents:
            await message.answer(_NO_DOCUMENTS_TEXT)
            return
        await message.answer(self._corpus_text(documents))

    async def handle_delete(self, message: Message, command: CommandObject) -> None:
        """Команда /delete: удалить документ по имени, без подтверждения."""
        if self._chat_not_allowed(message):
            return
        owner_id = await self._owner_or_answer(message)
        if owner_id is None:
            return
        name = (command.args or "").strip()
        if not name:
            await message.answer(_DELETE_USAGE_TEXT)
            return
        try:
            self._documents.delete_document(owner_id, name)
        except DocumentNotFoundError:
            await message.answer(self._not_found_text(owner_id, name))
            return
        except ApplicationError:
            self._logger.error("document_delete_failed", chat_id=message.chat.id, status="error")
            await message.answer(_ERROR_TEXT)
            return
        self._logger.info("document_deleted", owner_id=int(owner_id))
        await message.answer(_DOCUMENT_DELETED_TEXT.format(name=name))

    async def _owner_or_answer(self, message: Message) -> TelegramUserId | None:
        """Владелец корпуса; без автора — понятная ошибка и ``None`` (ADR-0002)."""
        if message.from_user is None:
            self._logger.info("command_without_author", chat_id=message.chat.id)
            await message.answer(_NO_AUTHOR_TEXT)
            return None
        return TelegramUserId(message.from_user.id)

    @staticmethod
    def _corpus_text(documents: Sequence[DocumentInfo]) -> str:
        """Корпус в одну строку-список для /documents и ошибки /delete."""
        listing = "\n".join(f"• {document.name}" for document in documents)
        return f"{_DOCUMENTS_HEADER_TEXT}\n{listing}"

    def _not_found_text(self, owner_id: TelegramUserId, name: str) -> str:
        """Дружелюбная ошибка удаления: имя не найдено — с текущим корпусом."""
        try:
            documents = self._documents.list_documents(owner_id)
        except ApplicationError:
            return _DOCUMENT_NOT_FOUND_TEXT.format(name=name)
        if not documents:
            return f"{_DOCUMENT_NOT_FOUND_TEXT.format(name=name)}\n{_NO_DOCUMENTS_TEXT}"
        return f"{_DOCUMENT_NOT_FOUND_TEXT.format(name=name)}\n{self._corpus_text(documents)}"


def _indexing_error_text(exc: ApplicationError) -> str:
    """Понятное сообщение для прикладной ошибки индексации (без деталей)."""
    if isinstance(exc, UnsupportedDocumentError):
        return "⚠ Такой формат не поддерживается. Пришлите документ .txt, .md, .pdf или .docx."
    if isinstance(exc, CorruptedDocumentError):
        return "⚠ Не удалось прочитать документ: файл повреждён или защищён паролем."
    if isinstance(exc, EmptyDocumentError):
        return "⚠ В документе не оказалось текста — индексировать нечего."
    if isinstance(exc, DocumentTooLargeError):
        return (
            "⚠ Документ слишком большой. Лимиты: файл до 20 MB, "
            "текст до 200 000 символов, до 300 чанков."
        )
    if isinstance(exc, EmbeddingError):
        return "⚠ Эмбеддинг-модель недоступна. Попробуйте загрузить документ позже."
    if isinstance(exc, RagError):
        return "⚠ Хранилище документов сейчас недоступно. Попробуйте позже."
    return _ERROR_TEXT
