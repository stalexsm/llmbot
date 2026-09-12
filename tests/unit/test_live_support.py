"""Юнит-тесты хелперов живых тестов: модели сервера Ollama и загрузка .env."""

import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.live.support import load_dotenv_into_environ, ollama_models


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)


async def test_ollama_models_returns_names_from_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"models": [{"name": "qwen3:4b"}, {"name": "bge-m3:latest"}]}
        )

    async with _client(handler) as client:
        models = await ollama_models(client, "http://localhost:11434")

    assert models == ("qwen3:4b", "bge-m3:latest")


async def test_ollama_models_none_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        assert await ollama_models(client, "http://localhost:11434") is None


async def test_ollama_models_none_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with _client(handler) as client:
        assert await ollama_models(client, "http://localhost:11434") is None


async def test_ollama_models_none_on_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    async with _client(handler) as client:
        assert await ollama_models(client, "http://localhost:11434") is None


async def test_ollama_models_trailing_slash_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"  # двойной слэш не допускается
        return httpx.Response(200, json={"models": []})

    async with _client(handler) as client:
        assert await ollama_models(client, "http://localhost:11434/") == ()


def test_load_dotenv_fills_missing_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# комментарий\n"
        "\n"
        "OLLAMA_MODEL=qwen3:4b\n"
        "OLLAMA_JUDGE_MODEL = 'qwen3:8b' \n"
        'TELEGRAM_BOT_TOKEN="123:x"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    load_dotenv_into_environ(env_file)

    assert os.environ["OLLAMA_MODEL"] == "qwen3:4b"
    assert os.environ["OLLAMA_JUDGE_MODEL"] == "qwen3:8b"
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "123:x"


def test_load_dotenv_never_overrides_real_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODEL", "from-environ")

    load_dotenv_into_environ(env_file)

    assert os.environ["OLLAMA_MODEL"] == "from-environ"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    load_dotenv_into_environ(tmp_path / "absent.env")

    assert "OLLAMA_MODEL" not in os.environ
