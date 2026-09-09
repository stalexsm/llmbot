"""Тесты статус-месседжа индексации: троттлинг правок и фолбэк."""

from unittest.mock import AsyncMock

import structlog.stdlib
from aiogram import Bot
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import Message
from pytest import MonkeyPatch

from bot.application.progress import IndexingStage
from bot.telegram.progress import TelegramIndexProgress
from tests.fakes import make_telegram_message


def make_progress(
    bot: Bot,
    monkeypatch: MonkeyPatch,
    logger: structlog.stdlib.BoundLogger,
    *,
    min_interval: float,
) -> tuple[TelegramIndexProgress, AsyncMock]:
    next_id = {"value": 100}

    async def fake_request(_bot: Bot, method: object, **_kwargs: object) -> Message | None:
        if isinstance(method, SendMessage):
            next_id["value"] += 1
            return Message.model_validate(
                {
                    "message_id": next_id["value"],
                    "date": 1735689600,
                    "chat": {"id": 100, "type": "private"},
                    "text": method.text,
                }
            ).as_(_bot)
        return None

    request_mock = AsyncMock(side_effect=fake_request)
    monkeypatch.setattr(bot.session, "make_request", request_mock)
    status = make_telegram_message("статус").as_(bot)
    return TelegramIndexProgress(
        status, "doc.txt", logger, min_interval_seconds=min_interval
    ), request_mock


def edits(request_mock: AsyncMock) -> list[EditMessageText]:
    return [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], EditMessageText)
    ]


async def test_stages_within_interval_are_throttled(
    bot: Bot, monkeypatch: MonkeyPatch, logger: structlog.stdlib.BoundLogger
) -> None:
    progress, request_mock = make_progress(bot, monkeypatch, logger, min_interval=100.0)

    await progress.on_stage(IndexingStage.EXTRACTING)
    await progress.on_stage(IndexingStage.CHUNKED, chunks=3)
    await progress.on_stage(IndexingStage.EMBEDDING, chunks=3)

    # Первая правка проходит, соседние по времени схлопываются.
    assert len(edits(request_mock)) == 1


async def test_final_result_is_always_shown(
    bot: Bot, monkeypatch: MonkeyPatch, logger: structlog.stdlib.BoundLogger
) -> None:
    progress, request_mock = make_progress(bot, monkeypatch, logger, min_interval=100.0)

    await progress.on_stage(IndexingStage.EXTRACTING)
    await progress.finish_success(3)

    texts = [edit.text for edit in edits(request_mock)]
    assert len(texts) == 2
    assert "Документ готов" in (texts[-1] or "")


async def test_edit_failure_falls_back_to_new_message(
    bot: Bot, monkeypatch: MonkeyPatch, logger: structlog.stdlib.BoundLogger
) -> None:
    from aiogram.exceptions import TelegramBadRequest

    progress, _unused_mock = make_progress(bot, monkeypatch, logger, min_interval=0.0)

    async def failing_edit(_bot: Bot, method: object, **_kwargs: object) -> Message | None:
        if isinstance(method, EditMessageText):
            raise TelegramBadRequest(method=method, message="message is not modified")
        return None

    failing_mock = AsyncMock(side_effect=failing_edit)
    monkeypatch.setattr(bot.session, "make_request", failing_mock)

    await progress.finish_error("что-то сломалось")

    # Правка невозможна — вместо неё ушло новое сообщение.
    methods = [call.args[1] for call in failing_mock.await_args_list]
    assert any(isinstance(method, SendMessage) for method in methods)
    sent = [method for method in methods if isinstance(method, SendMessage)][-1]
    assert "сломалось" in (sent.text or "")
