"""Каркас живых тестов: маркер live, загрузка .env и фикстура живого Ollama.

Каждый тест под ``tests/live`` автоматически получает маркер ``live`` —
дефолтный прогон (``-m "not live"``) его не запускает; явный запуск —
``uv run pytest -m live``. Без работающего Ollama живой тест скипается,
а не падает: проверка доступности общая, копипасты нет.
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from tests.live.support import LiveOllama, load_dotenv_into_environ, ollama_models

_LIVE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _LIVE_ROOT.parent.parent
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen3:4b"
_DEFAULT_NUM_CTX = 8192
_DEFAULT_TIMEOUT_SECONDS = 120.0
_TRUTHY = {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Все тесты под tests/live — живые: отсечка дефолтным прогоном по маркеру."""
    for item in items:
        if _LIVE_ROOT in item.path.parents:
            item.add_marker(pytest.mark.live)


@pytest.fixture
async def live_ollama() -> AsyncIterator[LiveOllama]:
    """Живой Ollama по настройкам окружения; недоступен — тест скипается."""
    # .env подгружается в фикстуре, а не на импорте conftest: мутация
    # os.environ не задевает офлайн-тесты дефолтного прогона; приоритет —
    # у настоящего экспорта.
    load_dotenv_into_environ(_REPO_ROOT / ".env")
    # Пустое значение считаем не заданным: живой тест скипается по
    # доступности сервера, а не падает на парсинге пустой строки из .env.
    base_url = os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_BASE_URL
    timeout_seconds = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS") or _DEFAULT_TIMEOUT_SECONDS)
    probe = httpx.AsyncClient(timeout=2.0, trust_env=False)
    try:
        models = await ollama_models(probe, base_url)
    finally:
        await probe.aclose()
    if models is None:
        pytest.skip(f"Ollama недоступен по адресу {base_url}: живой тест пропущен")

    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), trust_env=False)
    try:
        yield LiveOllama(
            client=client,
            base_url=base_url,
            models=models,
            model=os.environ.get("OLLAMA_MODEL") or _DEFAULT_MODEL,
            think=os.environ.get("OLLAMA_THINK", "").strip().lower() in _TRUTHY,
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX") or _DEFAULT_NUM_CTX),
            timeout_seconds=timeout_seconds,
        )
    finally:
        await client.aclose()
