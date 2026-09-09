"""Test doubles and factories shared across unit and integration tests."""

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import structlog.stdlib
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Message
from pytest import MonkeyPatch

from bot.domain.ids import ModelId, RequestId, ToolId
from bot.domain.messages import InferenceMessage
from bot.domain.tools import ToolCall
from bot.inference.embeddings import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from bot.inference.models import InferenceRequest, InferenceResponse, InferenceUsage
from bot.rag.migrations import apply_migrations as apply_rag_migrations
from bot.rag.models import EMBEDDING_DIMENSION
from bot.rag.service import RagService
from bot.rag.store import RagStore


class RecordingProgress:
    """Прогресс, записывающий события шагов агента для проверок."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, bool | None]] = []

    async def command_started(self, command: str) -> None:
        self.events.append(("started", command, None))

    async def command_finished(self, command: str, succeeded: bool) -> None:
        self.events.append(("finished", command, succeeded))


class MockInferenceProvider:
    """Deterministic in-memory inference provider for tests.

    Structurally implements ``bot.inference.provider.InferenceProvider``.
    """

    def __init__(
        self,
        response_content: str = "Ответ модели",
        usage: InferenceUsage | None = None,
    ) -> None:
        self.response_content = response_content
        self.usage = usage
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(
            request_id=request.request_id,
            content=self.response_content,
            usage=self.usage,
        )


class ScriptedInferenceProvider:
    """Провайдер с заранее заготовленными ответами по порядку (для агентного цикла)."""

    def __init__(self, responses: list[InferenceResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("scripted inference responses are exhausted")
        return self._responses.pop(0)


def tool_call_arguments(command: str) -> str:
    """JSON-аргументы вызова exec-инструмента для скриптованных ответов."""
    return json.dumps({"command": command}, ensure_ascii=False)


def exec_call_response(*commands: str) -> InferenceResponse:
    """Ответ модели, запрашивающий инструмент exec с указанными командами."""
    return InferenceResponse(
        request_id=RequestId("scripted"),
        content="",
        tool_calls=tuple(
            ToolCall(
                name=ToolId("execute_command"),
                arguments=json.dumps({"command": command}, ensure_ascii=False),
            )
            for command in commands
        ),
    )


def tool_call_response(name: str, arguments: str) -> InferenceResponse:
    """Ответ модели с произвольным вызовом инструмента."""
    return InferenceResponse(
        request_id=RequestId("scripted"),
        content="",
        tool_calls=(ToolCall(name=ToolId(name), arguments=arguments),),
    )


def final_response(text: str) -> InferenceResponse:
    """Финальный ответ модели без вызовов инструментов."""
    return InferenceResponse(request_id=RequestId(str(uuid4())), content=text)


class StubQueryRewriter:
    """Фейковка порта rag-слоя QueryRewriter: заготовка, эхо или сбой.

    Без заготовки возвращает вопрос как есть (пасsthrough); каждый вызов
    запоминает (request_id, вопрос, реплики) для проверок прокидывания.
    """

    def __init__(self, rewritten: str | None = None, *, error: Exception | None = None) -> None:
        self.rewritten = rewritten
        self.error = error
        self.calls: list[tuple[RequestId, str, tuple[str, ...]]] = []

    async def rewrite(
        self,
        request_id: RequestId,
        question: str,
        recent_turns: tuple[str, ...],
    ) -> str:
        self.calls.append((request_id, question, recent_turns))
        if self.error is not None:
            raise self.error
        return self.rewritten if self.rewritten is not None else question


class FailingInferenceProvider:
    """Inference provider that always raises the configured exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise self.error


class SpyMetricsCollector:
    """Фальшивка RunMetrics: запоминает закрытые запуски для проверок."""

    def __init__(self) -> None:
        self.finished: list[tuple[RequestId, bool]] = []

    def record_llm_call(
        self,
        *,
        request_id: RequestId,
        model: ModelId,
        latency_ms: int,
        usage: InferenceUsage | None,
        messages: tuple[InferenceMessage, ...],
        reached_model: bool,
    ) -> None:
        pass

    def finish_run(self, request_id: RequestId, *, success: bool) -> None:
        self.finished.append((request_id, success))


def make_telegram_message(
    text: str,
    *,
    message_id: int = 42,
    chat_id: int = 100,
    user_id: int = 7,
    with_author: bool = True,
) -> Message:
    """Build a realistic private-chat aiogram Message without any I/O."""
    data: dict = {
        "message_id": message_id,
        "date": 1735689600,
        "chat": {"id": chat_id, "type": "private"},
        "text": text,
    }
    if with_author:
        data["from"] = {"id": user_id, "is_bot": False, "first_name": "Tester"}
    return Message.model_validate(data)


class MockEmbeddingProvider:
    """Детерминированный in-memory провайдер эмбеддингов для тестов.

    Структурно реализует ``bot.inference.embeddings.EmbeddingProvider``:
    вектор текста — устойчивая функция его байтов (одинаковый текст даёт
    одинаковый вектор). Семантическую близость реальных моделей фейковка
    не воспроизводит — пороги в тестах задаются явно.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.requests: list[EmbeddingRequest] = []
        self._dimension = dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(
            request_id=request.request_id,
            embeddings=tuple(self._vector(text) for text in request.texts),
        )

    def _vector(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimension
        for position, byte in enumerate(text.encode("utf-8")[: self._dimension]):
            values[position] = (byte % 32) / 31.0
        return tuple(values)


class KeywordEmbeddingProvider:
    """Фейковка эмбеддингов с семантикой «одно ключевое слово — одна ось».

    Текст получает one-hot вектор по первому известному ключевому слову:
    запрос и чанки с одним словом дают косинусную близость 1, текст без
    известных слов уходит на отдельную ось-заглушку и ни с чем не совпадает.
    Поиск по порогу становится детерминированным без живой модели.
    """

    def __init__(self, keywords: tuple[str, ...], dimension: int = EMBEDDING_DIMENSION) -> None:
        if len(keywords) + 1 > dimension:
            raise ValueError("keywords must fit the embedding dimension")
        self.requests: list[EmbeddingRequest] = []
        self._keywords = keywords
        self._dimension = dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(
            request_id=request.request_id,
            embeddings=tuple(self._vector(text) for text in request.texts),
        )

    def _axis(self, text: str) -> int:
        lowered = text.lower()
        for index, keyword in enumerate(self._keywords):
            if keyword in lowered:
                return index
        return len(self._keywords)  # ось «мимо корпуса»: ни с чем не совпадает

    def _vector(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimension
        values[self._axis(text)] = 1.0
        return tuple(values)


# Ключевые слова по умолчанию для тестового поиска: «отпуск» — ось 0,
# «командировка» — ось 1; тексты без них ни с чем не совпадают.
_TEST_KEYWORDS = ("отпуск", "командировка")


def make_rag_service(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    embeddings: EmbeddingProvider | None = None,
) -> RagService:
    """RagService с tmp-БД и настройками по умолчанию (порог 0.5)."""
    database = tmp_path / "rag.db"
    apply_rag_migrations(database)
    return RagService(
        store=RagStore(database=database, logger=logger),
        embeddings=embeddings or KeywordEmbeddingProvider(_TEST_KEYWORDS),
        model=ModelId("fake-embed"),
        chunk_target_chars=900,
        chunk_overlap_chars=150,
        top_k=5,
        overfetch=4,
        min_similarity=0.5,
        max_file_bytes=20 * 1024 * 1024,
        max_text_chars=200_000,
        max_chunks=300,
        logger=logger,
    )


def mock_telegram_api(bot: Bot, monkeypatch: MonkeyPatch) -> AsyncMock:
    """Перехват исходящих Telegram API вызовов без сети.

    SendMessage возвращает свежее сообщение, привязанное к боту: правки
    статуса через ``edit_text`` в тестах работают, как в живом Telegram.
    """
    next_id = {"value": 100}

    async def fake_request(bot: Bot, method: object, **_kwargs: object) -> Message | None:
        if isinstance(method, SendMessage):
            next_id["value"] += 1
            return Message.model_validate(
                {
                    "message_id": next_id["value"],
                    "date": 1735689600,
                    "chat": {"id": 100, "type": "private"},
                    "text": method.text,
                }
            ).as_(bot)
        return None

    request_mock = AsyncMock(side_effect=fake_request)
    monkeypatch.setattr(bot.session, "make_request", request_mock)
    return request_mock
