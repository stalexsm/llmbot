"""Shared pytest fixtures."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import structlog
import structlog.stdlib
from aiogram import Bot
from structlog.testing import CapturingLogger


@pytest.fixture
def repo_root() -> Path:
    """Корень репозитория: фикстура для чтения настоящих файлов проекта."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def capturing_logger() -> CapturingLogger:
    return CapturingLogger()


@pytest.fixture
def logger(capturing_logger: CapturingLogger) -> structlog.stdlib.BoundLogger:
    """Quiet structlog logger that records events for later assertions."""
    return structlog.stdlib.BoundLogger(capturing_logger, [], {})


@pytest.fixture
async def bot() -> AsyncIterator[Bot]:
    """aiogram Bot with a real session; network calls are mocked per test."""
    bot = Bot(token="123456:TEST-TOKEN")
    try:
        yield bot
    finally:
        await bot.session.close()
