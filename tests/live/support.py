"""Общая поддержка живых тестов: окружение и доступность Ollama.

Живые тесты (маркер ``live``) гоняют настоящую модель против работающего
сервера Ollama и читают те же переменные окружения, что и бот, — напрямую,
без pydantic-settings. Если сервер недоступен, живые тесты скипаются:
гейты остаются зелёными без запущенной модели.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import httpx


def load_dotenv_into_environ(path: Path) -> None:
    """Подложить переменные из .env в окружение, не перекрывая уже заданные.

    Настоящий экспорт в окружении имеет приоритет над файлом — как у
    pydantic-settings; значения в кавычках очищаются от них.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def ollama_models(client: httpx.AsyncClient, base_url: str) -> tuple[str, ...] | None:
    """Имена моделей на сервере Ollama (GET /api/tags).

    ``None`` — сервер недоступен, отвечает ошибкой или невалидным телом:
    живой тест в этом случае скипается, а не падает.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = await client.get(url)
        if response.status_code != httpx.codes.OK:
            return None
        return tuple(str(model.get("name", "")) for model in response.json().get("models", []))
    except httpx.HTTPError:
        return None
    except ValueError:  # тело не JSON
        return None


@dataclass(frozen=True)
class LiveOllama:
    """Подключение к живому серверу Ollama по настройкам окружения.

    Клиент принадлежит фикстуре ``live_ollama`` и закрывается ею же;
    модели — имена из /api/tags на момент проверки доступности.
    """

    client: httpx.AsyncClient
    base_url: str
    models: tuple[str, ...]
    model: str
    think: bool
    num_ctx: int
    timeout_seconds: float
