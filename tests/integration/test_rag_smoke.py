"""Живой smoke RAG-ядра против Ollama с bge-m3.

Требует запущенный Ollama и скачанную модель эмбеддингов
(``ollama pull bge-m3``); без сервера или без модели тест пропускается —
гейты остаются зелёными офлайн. Проверяет полный путь тикета 02:
реальные эмбеддинги → rag-БД (tmp-файл) → поиск с порогом близости.
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import structlog
import structlog.stdlib

from bot.domain.ids import ModelId, RequestId, TelegramUserId
from bot.inference.embeddings_ollama import OllamaEmbeddingProvider
from bot.rag.migrations import apply_migrations
from bot.rag.service import RagService
from bot.rag.store import RagStore

_OWNER = TelegramUserId(42)
_REQUEST_ID = RequestId("rag-smoke")

_DOCUMENT = (
    "Регламент отпусков\n"
    "\n"
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. Отпуск можно\n"
    "разделить на части, если хотя бы одна из них не короче 14 дней.\n"
    "\n"
    "Порядок согласования\n"
    "\n"
    "Заявление на отпуск подается не позднее чем за две недели до его начала\n"
    "и согласовывается с руководителем отдела.\n"
    "\n"
    "Командировки\n"
    "\n"
    "Суточные при командировке по России выплачиваются в размере 700 рублей\n"
    "за каждый день, включая дни в пути.\n"
    "\n"
    "Техника безопасности\n"
    "\n"
    "Перед началом работы сотрудник обязан пройти инструктаж на рабочем месте\n"
    "и расписаться в журнале регистрации инструктажа.\n"
    "\n"
    "Удалённая работа\n"
    "\n"
    "Дистанционный формат оформляется дополнительным соглашением к трудовому\n"
    "договору. Рабочее время сотрудника на удалёнке совпадает с офисным графиком.\n"
    "\n"
    "Материальная ответственность\n"
    "\n"
    "За оборудование, выданное сотруднику, он несёт полную материальную\n"
    "ответственность до момента возврата имущества на склад.\n"
    "\n"
    "Больничные листы\n"
    "\n"
    "Электронный больничный лист передаётся в отдел кадров автоматически;\n"
    "пособие начисляется в течение десяти рабочих дней после закрытия.\n"
    "\n"
    "Обучение и аттестация\n"
    "\n"
    "Каждый сотрудник проходит ежегодную аттестацию по своему профилю;\n"
    "результаты влияют на пересмотр заработной платы по итогам года."
)

_RELEVANT_QUERY = "сколько дней отпуска положено сотруднику"
_IRRELEVANT_QUERY = "расписание поездов до Владивостока"


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture
async def embeddings(
    logger: structlog.stdlib.BoundLogger,
) -> AsyncIterator[tuple[OllamaEmbeddingProvider, httpx.AsyncClient]] | None:
    """Живой провайдер эмбеддингов; без Ollama или без bge-m3 — пропуск."""
    url = _base_url()
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            response = await client.get(f"{url}/api/tags")
            models = [model.get("name", "") for model in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        pytest.skip(f"Ollama is not reachable at {url}")
    if not any(name.startswith("bge-m3") for name in models):
        pytest.skip("bge-m3 model is not pulled: run `ollama pull bge-m3`")
    # Дефолт настройки OLLAMA_EMBED_TIMEOUT_SECONDS: живая модель на
    # локальной машине укладывается с запасом.
    http_client = httpx.AsyncClient(trust_env=False)
    provider = OllamaEmbeddingProvider(
        client=http_client,
        base_url=url,
        timeout_seconds=120.0,
        logger=logger,
    )
    try:
        yield provider, http_client
    finally:
        await http_client.aclose()


async def test_live_index_and_search_with_bge_m3(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    embeddings: tuple[OllamaEmbeddingProvider, httpx.AsyncClient] | None,
) -> None:
    assert embeddings is not None  # гарантия фикстуры после skip
    provider, _http_client = embeddings
    database = tmp_path / "rag.db"
    apply_migrations(database)
    service = RagService(
        store=RagStore(database=database, logger=logger),
        embeddings=provider,
        model=ModelId("bge-m3"),
        chunk_target_chars=900,
        chunk_overlap_chars=150,
        top_k=5,
        overfetch=4,
        min_similarity=0.35,
        max_file_bytes=20 * 1024 * 1024,
        max_text_chars=200_000,
        max_chunks=300,
        logger=logger,
    )

    indexed = await service.index_document(_REQUEST_ID, _OWNER, "reglament.txt", _DOCUMENT.encode())

    assert indexed.chunk_count >= 2
    relevant = await service.search(_REQUEST_ID, _OWNER, _RELEVANT_QUERY)
    irrelevant = await service.search(_REQUEST_ID, _OWNER, _IRRELEVANT_QUERY)

    # Релевантный запрос находит чанки с отпуском; посторонний запрос
    # даёт худшую близость (порог калибруется в тикете 07 на датасете).
    assert relevant
    assert any("отпуск" in hit.text.lower() for hit in relevant)
    top_relevant = max(hit.similarity for hit in relevant)
    top_irrelevant = max((hit.similarity for hit in irrelevant), default=0.0)
    assert top_relevant > top_irrelevant
    assert all(hit.document_name == "reglament.txt" for hit in relevant)
