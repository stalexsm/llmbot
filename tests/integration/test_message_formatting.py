"""Интеграционный тест отправки: битая разметка во входе и ответе модели.

Полный офлайн-граф (handlers → application → agent → фейковый инференс,
чат-сессии во временной БД, без сети): спецсимволы и битая разметка в любом
тексте не ломают отправку — все части уходят экранированными с MarkdownV2
и в лимите Telegram.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import structlog.stdlib
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.methods import SendMessage
from pytest import MonkeyPatch

from bot.agent.loop import AgentLoop
from bot.application.service import ApplicationService
from bot.domain.ids import ModelId
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers
from tests.fakes import (
    ScriptedInferenceProvider,
    SpyMetricsCollector,
    final_response,
    make_telegram_message,
    mock_telegram_api,
)
from tests.unit.telegram.markdown_v2 import assert_valid_markdown_v2

# Ответ модели из одних спецсимволов и битой разметки: незакрытый код-блок,
# незакрытый код-спан, скобки без пары — всё, чем Telegram давит sendMessage.
BROKEN_RESPONSE = (
    "_жирный_ [ссылка](битая ```незакрытый блок\n2 * 2 = 4. `обрыв\n> | { } ! = + # - ~ ( ) ."
)

BROKEN_INPUT = "Привет *мир*! [битая](разметка _и снова_"


def make_stack(
    logger: structlog.stdlib.BoundLogger,
    provider: ScriptedInferenceProvider,
    tmp_path: Path,
) -> TelegramHandlers:
    """Граф текстового сообщения без RAG и exec: инференс по сценарию."""
    loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt="Ты тестовый агент.",
        tools=(),
        step_limit=10,
        logger=logger,
    )
    chat_database = tmp_path / "chats.db"
    apply_migrations(chat_database)
    service = ApplicationService(
        agent=loop,
        sessions=ChatSessionStore(database=chat_database, logger=logger),
        history_limit=20,
        logger=logger,
        metrics=SpyMetricsCollector(),
    )
    return TelegramHandlers(
        service=service,
        documents=AsyncMock(),  # документы в сценарии не участвуют
        document_loader=AsyncMock(),
        logger=logger,
        allowed_chat_ids=frozenset(),
    )


def sent_messages(request_mock: AsyncMock) -> list[SendMessage]:
    """Все исходящие SendMessage, перехваченные моком сессии."""
    return [
        call.args[1]
        for call in request_mock.await_args_list
        if isinstance(call.args[1], SendMessage)
    ]


async def test_broken_markup_in_input_and_response_is_sent_without_error(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = ScriptedInferenceProvider([final_response(BROKEN_RESPONSE)])
    handlers = make_stack(logger, provider, tmp_path)
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_text(make_telegram_message(BROKEN_INPUT).as_(bot))

    # Цикл завершился и ответ доставлен: ни одного исключения, все части
    # с MarkdownV2, в лимите и с валидной разметкой.
    sent = sent_messages(request_mock)
    assert sent
    assert all(part.parse_mode == ParseMode.MARKDOWN_V2 for part in sent)
    assert all(len(part.text or "") <= 4096 for part in sent)
    for part in sent:
        assert_valid_markdown_v2(part.text or "")
    # Экранирование — только на исходящей стороне: модель видит ввод как есть.
    assert provider.requests[0].messages[-1].content == BROKEN_INPUT
    # Ответ целиком дошёл до пользователя: без слешей экранирования склеивается
    # в исходный текст.
    assert "".join(part.text for part in sent).replace("\\", "") == BROKEN_RESPONSE
